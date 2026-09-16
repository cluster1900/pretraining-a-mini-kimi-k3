"""Synthetic stage-regression fixture using the real, pinned tokenizer.

This report does not certify the production corpus or full model training.
"""
import argparse
import json
import random
import string
import tempfile
from pathlib import Path
from canonical_v2 import SPECS,atomic_json,digest_file
from stage_audit import BENCHMARKS
from clean_v2 import clean_record,clean_source
from dedup_v2 import SOURCES,group_key,assigned_split,deduplicate
from near_dedup_v2 import near_deduplicate,features,similarity,bands
from contamination_v2 import compile_index,filter_source
from tokenize_v2 import encode_source
from finalize_v2 import finalize
from train.config import MiniK3Config
from train.data.smoke_audit import verified_smoke_manifest


def run(tokenizer,output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pipeline-fixture-',dir=output.parent) as td:
        root=Path(td);rng=random.Random(345)
        for source in SOURCES:
            selected={}
            for candidate in range(10000):
                text=' '.join(''.join(rng.choices(string.ascii_lowercase,k=7)) for _ in range(60))
                row={'id':source+str(candidate),'source':source,'repo':'fixture'}
                if source in ('openassistant','openhermes','openr1'):
                    row.update(kind='sft',messages=[{'role':'user','content':text},{'role':'assistant','content':'The requested answer [EOS] is literal text.'}])
                elif source=='ultrafeedback':
                    row.update(kind='preference',prompt=[{'role':'user','content':text}],chosen='Preferred answer',rejected='Different rejected answer')
                else:row.update(kind='code' if source=='code-python' else 'text',text=text,license_type='MIT')
                cleaned,reason=clean_record(row)
                if reason:raise AssertionError(reason)
                split=assigned_split(group_key(cleaned))
                selected.setdefault(split,row)
                if len(selected)==2:break
            if len(selected)!=2:raise AssertionError('Fixture cannot create both splits')
            d=root/'canonical'/source;d.mkdir(parents=True)
            p=d/'part-00000.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in selected.values()))
            atomic_json(d/'COMPLETE.json',{'source':source,'status':'complete','repo':SPECS[source][1],
                'counts':{'output_records':2},'script_sha256':digest_file(Path(__file__).with_name('canonical_v2.py')),
                'parts':[{'path':str(p),'documents':2,'bytes':p.stat().st_size,'sha256':digest_file(p)}]})
            clean_source(source,root)
        atomic_json(root/'SOURCE_REVIEW.json', {'status':'passed','mode':'synthetic', 'sources': {
            s: {'training_approved': True, 'license_review_status': 'passed', 'evidence': ['synthetic-fixture']}
            for s in SOURCES}})
        x=features(' '.join('hello'+str(i) for i in range(100)))
        assert similarity(x,x)==1 and bands(x)==bands(x)
        deduplicate(root);near_deduplicate(root)
        benchmark=root/'benchmarks.jsonl';benchmark.write_text(''.join(json.dumps({'benchmark':name,'segments':[
            'unique unrelated benchmark one two three four five six seven eight nine ten eleven twelve thirteen']})+'\n'
            for name in sorted(BENCHMARKS)))
        index=root/'benchmarks/13grams.json.gz';compile_index(benchmark,index)
        for source in SOURCES:
            filter_source(source,root,index)
            encode_source(source,root,tokenizer,shard_tokens=128)
        report=finalize(root)
        assert report['train_validation_overlap']==0 and len(report['sources'])==11
        cfg=MiniK3Config()
        verified_smoke_manifest(root/'manifests/pretrain_stable.json',cfg.stable_mix,cfg.vocab_size)
        output.write_text(json.dumps({'status':'passed','scope':'synthetic regression with real tokenizer',
            'sources':11,'stages':['canonical fixture','clean','global exact','MinHash LSH','13gram','encode','shards','manifest audit'],
            'tokenizer_fingerprint':report['tokenizer_fingerprint']},indent=2))
        print('END_TO_END_REGRESSION_PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();run(a.tokenizer,a.report)
