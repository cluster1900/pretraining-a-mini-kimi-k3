"""Verified data-stage controller; never launches the full training run."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from canonical_v2 import SPECS,atomic_json
from dedup_v2 import SOURCES


def live(script,source=None):
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=entry.read_bytes().split(b'\0')
            if str(script).encode() in args and (source is None or source.encode() in args):return int(entry.parent.name)
        except OSError:pass
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data/mini-k3/data/prepared-v2');p.add_argument('--work',default='/data/mini-k3');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=Path(a.root);work=Path(a.work);scripts=work/'project/train/data';python=work/'venv/bin/python'
    logs=work/'logs/prepared-v2';logs.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
    state=root/'PIPELINE_STATUS.json';status={'status':'running','stage':'canonical'}
    def update(**kwargs):
        status.update(kwargs);atomic_json(state,dict(status,updated_at=time.time()))
    def run(script,args,logname):
        with (logs/logname).open('ab',buffering=0) as log:
            child=subprocess.Popen([str(python),'-u',str(script),*args],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            result=child.wait()
        if result:raise RuntimeError(f'{script.name} exit={result}; log={logs/logname}')
    def parallel(stage,script,args_for):
        update(stage=stage)
        def task(source):
            marker=root/stage/source/'COMPLETE.json'
            while live(script,source):
                if marker.exists():break
                time.sleep(15)
            if marker.exists():
                if json.loads(marker.read_text()).get('status')!='complete':raise ValueError('Invalid completion status')
                return
            if (root/stage/source/'PROGRESS.json').exists():
                raise RuntimeError(f'{source}: interrupted {stage} output needs recovery; not restarting silently')
            run(script,args_for(source),f'{stage}-{source}.log')
            if not marker.exists():raise RuntimeError(f'{source}: {stage} completion report missing')
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for future in as_completed([ex.submit(task,s) for s in SOURCES]):future.result()
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            update()
            while not all((root/'canonical'/s/'COMPLETE.json').exists() for s in SOURCES):
                if not any(live(scripts/'canonical_v2.py',s) for s in SOURCES) and not live(scripts/'run_canonical_v2.py'):
                    raise RuntimeError('Canonical conversion incomplete and no live converter')
                time.sleep(15)
            parallel('cleaned',scripts/'clean_v2.py',lambda s:['--source',s,'--root',str(root)])
            update(stage='global_exact_dedup')
            run(scripts/'dedup_v2.py',['--root',str(root)],'global-exact-dedup.log')
            update(stage='global_near_dedup')
            run(scripts/'near_dedup_v2.py',['--root',str(root)],'global-near-dedup.log')
            index=root/'benchmarks/13grams.json.gz'
            if not index.exists():raise RuntimeError('Verified benchmark index missing')
            parallel('decontaminated',scripts/'contamination_v2.py',lambda s:['filter','--source',s,'--root',str(root),'--index',str(index),'--input-stage','near-deduped'])
            update(stage='tokenization')
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futures=[ex.submit(run,scripts/'tokenize_v2.py',['--source',s,'--root',str(root),'--tokenizer',str(work/'data/tokenizer')],f'encode-{s}.log') for s in SOURCES]
                for f in as_completed(futures):f.result()
            update(stage='full_manifest_audit')
            run(scripts/'finalize_v2.py',['--root',str(root)],'final-manifest-audit.log')
            update(stage='model_smoke')
            report=root/'manifests/SMOKE.json'
            run(work/'project/train/smoke_from_manifest.py',['--manifest',str(root/'manifests/pretrain_stable.json'),'--report',str(report)],'model-smoke.log')
            if json.loads(report.read_text())['status']!='passed':raise RuntimeError('Model smoke not passed')
            update(status='complete',stage='complete')
        except Exception as exc:
            update(status='failed',error=str(exc));raise

if __name__=='__main__':main()
