"""Build train/decay/validation/alignment manifests only after full data audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from canonical_v2 import atomic_json,digest_file
from dedup_v2 import SOURCES
from stage_audit import verify_benchmark, verify_source_review, verify_stage_chain
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from train.config import MiniK3Config


def finalize(root):
    root=Path(root);config=MiniK3Config();config.validate();reports={}
    for source in SOURCES:
        path=root/'tokenized'/source/'COMPLETE.json'
        if not path.exists():raise ValueError('Tokenization incomplete: '+source)
        reports[source]=json.loads(path.read_text())
    source_review=verify_source_review(root)
    benchmark=verify_benchmark(root)
    chains={source:verify_stage_chain(root,source,benchmark=benchmark) for source in SOURCES}
    if any(r.get('vocab_size')!=config.vocab_size for r in reports.values()):
        raise ValueError('Token output vocabulary differs from model config')
    fps={r['tokenizer_fingerprint'] for r in reports.values()}
    if len(fps)!=1:raise ValueError('Tokenizer mismatch among sources')
    out=root/'manifests';out.mkdir(parents=True,exist_ok=True)
    auditdb=out/'split-audit.sqlite'
    if auditdb.exists():auditdb.unlink()
    db=sqlite3.connect(auditdb);db.execute('CREATE TABLE groups (key TEXT PRIMARY KEY, split TEXT)');db.execute('CREATE TABLE ids (id TEXT PRIMARY KEY)')
    summary={};total_tokens=0
    try:
        for source,r in reports.items():
            ledger=Path(r['document_index']['path'])
            if digest_file(ledger)!=r['document_index']['sha256']:raise ValueError('Document index corrupted')
            counts={'train':0,'validation':0}
            for line in ledger.open(encoding='utf-8'):
                row=json.loads(line);split=row['split'];counts[split]+=1
                db.execute('INSERT INTO ids VALUES (?)',(row['id'],))
                current=db.execute('SELECT split FROM groups WHERE key=?',(row['split_group'],)).fetchone()
                if current and current[0]!=split:raise ValueError('Train/validation group overlap')
                db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)',(row['split_group'],split))
            for split,n in counts.items():
                if n!=r['counts'].get(split+'_documents',0):raise ValueError('Document count mismatch')
            tokens={'train':0,'validation':0};eos={'train':0,'validation':0}
            for f in r['files']:
                p=Path(f['path'])
                if not p.is_file() or p.stat().st_size!=f['bytes'] or digest_file(p)!=f['sha256']:raise ValueError('Token output hash mismatch: '+str(p))
                split=f['split']
                if r['kind']=='pretrain':
                    if f['bytes']!=f['tokens']*4:raise ValueError('uint32 byte count mismatch')
                    with p.open('rb') as handle:
                        while True:
                            data=np.fromfile(handle,dtype='<u4',count=1_000_000)
                            if not len(data):break
                            if int(data.max())>=r['vocab_size']:raise ValueError('Out-of-range token ID')
                            tokens[split]+=len(data);eos[split]+=int((data==163585).sum())
                else:
                    records=0
                    for line in p.open(encoding='utf-8'):
                        row=json.loads(line);records+=1
                        if row['split']!=split:raise ValueError('Alignment split mismatch')
                        if r['kind']=='sft':
                            ids=row['input_ids'];labels=row['labels']
                            if len(ids)!=len(labels) or not any(x!=-100 for x in labels[1:]):raise ValueError('Invalid SFT targets')
                            if any(y!=-100 and y!=x for x,y in zip(ids,labels)):raise ValueError('Misaligned SFT labels')
                            sequences=[ids]
                        else:
                            n=row['prompt_len'];c=row['chosen_ids'];rej=row['rejected_ids']
                            if n<1 or n>=min(len(c),len(rej)) or c[:n]!=rej[:n]:raise ValueError('Invalid preference prompt boundary')
                            sequences=[c,rej]
                        for ids in sequences:
                            if not ids or any(t<0 or t>=r['vocab_size'] for t in ids):raise ValueError('Invalid alignment token ID')
                            tokens[split]+=len(ids)
                    if records!=counts[split]:raise ValueError('Alignment record count mismatch')
            for split in tokens:
                if tokens[split]!=r['counts'].get(split+'_tokens',0):raise ValueError('Token count mismatch')
                if r['kind']=='pretrain' and eos[split]!=counts[split]:raise ValueError('EOS/document mismatch')
            db.commit();summary[source]={'kind':r['kind'],'documents':counts,'tokens':tokens,'files':len(r['files'])}
            if r['kind']=='pretrain':total_tokens+=tokens['train']
            print(json.dumps({'source':source,'audit':'passed','tokens':tokens}),flush=True)
    finally:db.close()
    common={'schema_version':2,'tokenizer_fingerprint':next(iter(fps)),'vocab_size':config.vocab_size,'dtype':'<u4','status':'audited'}
    for filename,split,weights in [('pretrain_stable.json','train',config.stable_mix),('pretrain_decay.json','train',config.decay_mix),('validation.json','validation',config.stable_mix)]:
        data={'metadata':dict(common,split=split),'sources':{}}
        for source,weight in weights.items():
            r=reports[source];fs=[f for f in r['files'] if f['split']==split]
            if not fs:raise ValueError('Required source/split has no shard: '+source+'/'+split)
            data['sources'][source]={'weight':weight,'shards':[f['path'] for f in fs],
                'shard_metadata':fs,'total_tokens':summary[source]['tokens'][split],
                'total_docs':summary[source]['documents'][split]}
        atomic_json(out/filename,data)
    alignment={'metadata':common,'sources':{s:reports[s] for s in SOURCES if reports[s]['kind']!='pretrain'}}
    atomic_json(out/'alignment.json',alignment)
    report={'status':'passed','sources':summary,'pretrain_train_tokens':total_tokens,
        'stage_chains':chains,'stage_audit_script_sha256':digest_file(Path(__file__).with_name('stage_audit.py')),
        'benchmark_index_sha256':benchmark['index_sha256'],
        'source_review_sha256':digest_file(root/'SOURCE_REVIEW.json'),
        'manifest_sha256':{name:digest_file(out/name) for name in
            ('pretrain_stable.json','pretrain_decay.json','validation.json','alignment.json')},
        'train_validation_overlap':0,'tokenizer_fingerprint':next(iter(fps)),
        'script_sha256':digest_file(__file__),'scope':'All listed token files and document/group indexes scanned; model smoke separate',
        'warnings':['Available per-source tokens must be compared with the 10B training mixture before a long run']}
    atomic_json(out/'AUDIT.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();finalize(a.root)
