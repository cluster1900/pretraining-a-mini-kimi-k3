"""Verified data-stage controller; never launches the full training run."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from canonical_v2 import SPECS,atomic_json,digest_file
from dedup_v2 import SOURCES
from stage_audit import verify_benchmark, verify_source_review, verify_stage_chain


def live(script,source=None):
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=entry.read_bytes().split(b'\0')
            if str(script).encode() in args and (source is None or source.encode() in args):return int(entry.parent.name)
        except OSError:pass
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data/mini-k3/data/prepared-v2');p.add_argument('--work',default='/data/mini-k3');p.add_argument('--workers',type=int,default=2);p.add_argument('--log-dir',default=None)
    a=p.parse_args();root=Path(a.root);work=Path(a.work);scripts=work/'project/train/data';python=work/'venv/bin/python'
    if a.workers < 1:p.error('--workers must be positive')
    logs=Path(a.log_dir) if a.log_dir else work/'logs/prepared-v2';logs.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
    state=root/'PIPELINE_STATUS.json';status={'status':'running','stage':'source_review','pid':os.getpid(),'started_at':time.time()}
    archive=work/'project/train/reports/background'/f'{time.time_ns()}-{os.getpid()}'
    def update(**kwargs):
        status.update(kwargs);atomic_json(state,dict(status,updated_at=time.time()))
    def archive_reports(stage, paths=None):
        paths=paths or [root/stage/s/'COMPLETE.json' for s in SOURCES]
        index={}
        for path in paths:
            relative=path.relative_to(root);dest=archive/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
            index[str(relative)]={'sha256':digest_file(dest),'bytes':dest.stat().st_size}
        atomic_json(archive/(stage+'-INDEX.json'),index)
    def run(script,args,logname):
        with (logs/logname).open('ab',buffering=0) as log:
            child=subprocess.Popen([str(python),'-u',str(script),*args],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            print(json.dumps({'event':'child_started','script':script.name,'pid':child.pid,'log':str(logs/logname),'at':time.time()}),flush=True)
            result=child.wait()
        print(json.dumps({'event':'child_exited','script':script.name,'pid':child.pid,'exit_code':result,'at':time.time()}),flush=True)
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
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            update()
            # Acquire the lock first: a duplicate launcher must not overwrite
            # the live controller's status, even when approval verification fails.
            verify_source_review(root)
            verify_benchmark(root)
            archive_reports('approval',[root/'SOURCE_REVIEW.json'])
            code=archive/'code';code.mkdir(parents=True,exist_ok=True)
            for path in [*scripts.glob('*.py'),work/'project/train/config.py',work/'project/train/smoke_from_manifest.py']:
                shutil.copyfile(path,code/path.name)
            update(report_archive=str(archive))
            update(stage='canonical')
            while not all((root/'canonical'/s/'COMPLETE.json').exists() for s in SOURCES):
                if not any(live(scripts/'canonical_v2.py',s) for s in SOURCES) and not live(scripts/'run_canonical_v2.py'):
                    raise RuntimeError('Canonical conversion incomplete and no live converter')
                time.sleep(15)
            for source in SOURCES:verify_stage_chain(root,source,through='canonical')
            archive_reports('canonical')
            parallel('cleaned',scripts/'clean_v2.py',lambda s:['--source',s,'--root',str(root)])
            for source in SOURCES:verify_stage_chain(root,source,through='cleaned')
            archive_reports('cleaned')
            update(stage='global_exact_dedup')
            if not (all((root/'deduped'/s/'COMPLETE.json').exists() for s in SOURCES) and (root/'DEDUP_COMPLETE.json').exists()):
                run(scripts/'dedup_v2.py',['--root',str(root)],'global-exact-dedup.log')
            for source in SOURCES:verify_stage_chain(root,source,through='deduped')
            archive_reports('deduped')
            update(stage='global_near_dedup')
            if not (all((root/'near-deduped'/s/'COMPLETE.json').exists() for s in SOURCES) and (root/'NEAR_DEDUP_COMPLETE.json').exists()):
                run(scripts/'near_dedup_v2.py',['--root',str(root)],'global-near-dedup.log')
            for source in SOURCES:verify_stage_chain(root,source,through='near-deduped')
            archive_reports('near-deduped')
            index=root/'benchmarks/13grams.json.gz'
            if not index.exists():raise RuntimeError('Verified benchmark index missing')
            parallel('decontaminated',scripts/'contamination_v2.py',lambda s:['filter','--source',s,'--root',str(root),'--index',str(index),'--input-stage','near-deduped'])
            archive_reports('decontaminated')
            update(stage='tokenization')
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futures=[ex.submit(run,scripts/'tokenize_v2.py',['--source',s,'--root',str(root),'--tokenizer',str(work/'data/tokenizer')],f'encode-{s}.log') for s in SOURCES]
                for f in as_completed(futures):f.result()
            archive_reports('tokenized')
            update(stage='full_manifest_audit')
            run(scripts/'finalize_v2.py',['--root',str(root)],'final-manifest-audit.log')
            archive_reports('manifests',[root/'manifests'/name for name in ('pretrain_stable.json','pretrain_decay.json','validation.json','alignment.json','AUDIT.json')])
            update(stage='model_smoke')
            report=root/'manifests/SMOKE.json'
            run(work/'project/train/smoke_from_manifest.py',['--manifest',str(root/'manifests/pretrain_stable.json'),'--report',str(report)],'model-smoke.log')
            if json.loads(report.read_text())['status']!='passed':raise RuntimeError('Model smoke not passed')
            archive_reports('smoke',[report])
            update(status='complete',stage='complete')
        except Exception as exc:
            update(status='failed',error=str(exc));raise

if __name__=='__main__':main()
