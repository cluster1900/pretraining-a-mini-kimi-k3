"""Global exact deduplication with per-part transactional recovery and splits."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from canonical_v2 import atomic_json, digest_file

SOURCES=['openassistant','openhermes','openr1','ultrafeedback','code-python','fineweb-edu','chinese-fineweb-edu','cosmopedia','finemath','open-web-math','dolma-body']


def content_key(row):
    payload={k:row[k] for k in ('text','messages','prompt','chosen','rejected') if k in row}
    if not payload:raise ValueError('Missing document payload')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def group_key(row):
    if row.get('group_id'):value=['conversation',row['repo'],row['group_id']]
    elif row.get('repo_path'):value=['code_repo',row['repo_path']]
    elif row.get('url'):value=['url',row['url'].split('#',1)[0]]
    elif row['kind']=='sft':
        prompt=[]
        for turn in row['messages']:
            if turn['role']=='assistant':break
            prompt.append(turn)
        value=['prompt',prompt]
    elif row['kind']=='preference':value=['prompt',row['prompt']]
    else:value=['content',content_key(row)]
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def assigned_split(key,official_holdout=False):
    return 'validation' if official_holdout or int(key[:16],16)%100==0 else 'train'


def verify_part(item):
    path=Path(item['path'])
    if not path.is_file() or path.stat().st_size!=item['bytes'] or digest_file(path)!=item['sha256']:
        raise ValueError('Part size/hash mismatch: '+str(path))


def deduplicate(root,sources=None,after_part=None):
    root=Path(root);sources=sources or SOURCES
    markers={s:root/'cleaned'/s/'COMPLETE.json' for s in sources}
    for p in markers.values():
        if not p.is_file():raise ValueError('Cleaning incomplete: '+str(p))
    inputs={s:json.loads(p.read_text()) for s,p in markers.items()}
    identity={'algorithm':'global-exact-split-v2.1','sources':sources,
              'inputs':{s:digest_file(p) for s,p in markers.items()},'validation_modulus':100}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    lock_path=root/'dedup.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        con=sqlite3.connect(root/'dedup-index-v2.sqlite')
        con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=FULL')
        con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
        con.execute('CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, source TEXT, rowid_text TEXT)')
        con.execute('CREATE TABLE IF NOT EXISTS holdout (key TEXT PRIMARY KEY)')
        con.execute('CREATE TABLE IF NOT EXISTS parts (source TEXT, ordinal INTEGER, report TEXT, PRIMARY KEY(source,ordinal))')
        previous=con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if previous and previous[0]!=identity_hash:raise ValueError('Dedup inputs/policy changed; separate run required')
        con.execute("INSERT OR IGNORE INTO meta VALUES ('identity',?)",(identity_hash,));con.commit()
        # Read official validation records before accepting any training copies.
        if not con.execute("SELECT 1 FROM meta WHERE key='holdout_done'").fetchone():
            for source,up in inputs.items():
                if source!='openassistant':continue
                for part in up['parts']:
                    verify_part(part)
                    for line in Path(part['path']).open(encoding='utf-8'):
                        r=json.loads(line)
                        if r.get('official_split')=='validation':
                            con.executemany('INSERT OR IGNORE INTO holdout VALUES (?)',[(group_key(r),),(content_key(r),)])
            con.execute("INSERT INTO meta VALUES ('holdout_done','1')");con.commit()
        total=Counter();completed=[]
        try:
            for source in sources:
                dest=root/'deduped'/source;dest.mkdir(parents=True,exist_ok=True)
                counts=Counter();outparts=[]
                for ordinal,part in enumerate(inputs[source]['parts']):
                    saved=con.execute('SELECT report FROM parts WHERE source=? AND ordinal=?',(source,ordinal)).fetchone()
                    if saved:
                        report=json.loads(saved[0]);verify_part(report)
                    else:
                        verify_part(part);out=dest/f'part-{ordinal:05d}.jsonl';tmp=out.with_name(out.name+'.incomplete')
                        c=Counter();con.execute('BEGIN IMMEDIATE')
                        try:
                            with Path(part['path']).open(encoding='utf-8') as r,tmp.open('w',encoding='utf-8') as w:
                                for line in r:
                                    row=json.loads(line);key=content_key(row);c['input']+=1
                                    if key!=row.get('content_sha256'):raise ValueError('Record content hash mismatch')
                                    if not con.execute('INSERT OR IGNORE INTO seen VALUES (?,?,?)',(key,source,row['id'])).rowcount:
                                        c['duplicates']+=1;continue
                                    group=group_key(row)
                                    held=con.execute('SELECT 1 FROM holdout WHERE key IN (?,?) LIMIT 1',(group,key)).fetchone() is not None
                                    row['split_group']=group;row['split']=assigned_split(group,held)
                                    w.write(json.dumps(row,ensure_ascii=False)+'\n');c['kept']+=1;c[row['split']]+=1
                                w.flush();os.fsync(w.fileno())
                            os.replace(tmp,out)
                            report={'path':str(out),'documents':c['kept'],'bytes':out.stat().st_size,'sha256':digest_file(out),'counts':dict(c)}
                            con.execute('INSERT INTO parts VALUES (?,?,?)',(source,ordinal,json.dumps(report)))
                            if after_part:after_part(source,ordinal)
                            con.commit()
                        except BaseException:
                            con.rollback();raise
                    counts.update(report['counts']);outparts.append(report)
                    atomic_json(dest/'PROGRESS.json',{'source':source,'counts':dict(counts),'parts':len(outparts)})
                if counts['input']!=counts['kept']+counts['duplicates']:raise ValueError('Dedup counts do not balance')
                result={'source':source,'status':'complete','algorithm':'global-exact','counts':dict(counts),'parts':outparts,
                        'upstream_report_sha256':identity['inputs'][source],'global_identity':identity_hash,'script_sha256':digest_file(__file__)}
                atomic_json(dest/'COMPLETE.json',result);completed.append(source);total.update(counts)
                print(json.dumps({'source':source,'counts':dict(counts)}),flush=True)
            con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            atomic_json(root/'DEDUP_COMPLETE.json',{'status':'complete','sources':completed,'counts':dict(total),
                        'identity':identity,'identity_sha256':identity_hash,'database_sha256':digest_file(root/'dedup-index-v2.sqlite')})
        finally:con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data/mini-k3/data/prepared-v2')
    a=p.parse_args();deduplicate(a.root)
