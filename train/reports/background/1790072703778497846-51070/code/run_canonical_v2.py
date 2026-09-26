"""Bounded conversion supervisor; exit status and reports reflect failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
from pathlib import Path
import subprocess
import sys
from canonical_v2 import SPECS, atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='/data/mini-k3');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--sources',nargs='*',choices=sorted(SPECS));a=p.parse_args()
    base=Path(a.base);out=base/'data/prepared-v2/canonical';logs=base/'logs/prepared-v2';logs.mkdir(parents=True,exist_ok=True)
    sources=a.sources or list(SPECS)
    def work(source):
        marker=out/source/'COMPLETE.json'
        if marker.exists():
            raise RuntimeError(f'{source}: existing output needs explicit verification before reuse')
        status=logs/f'{source}.status.json'
        atomic_json(status,{'source':source,'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        with (logs/f'convert-{source}.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('canonical_v2.py')),'--source',source,
                '--root',str(base/'data/raw'),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
            atomic_json(status,{'source':source,'status':'running','pid':child.pid})
            result=child.wait()
        ok=result==0 and marker.exists()
        atomic_json(status,{'source':source,'status':'complete' if ok else 'failed','exit_code':result,'pid':child.pid})
        return source,ok
    failed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(work,s) for s in sources]):
            source,ok=fut.result()
            print(json.dumps({'source':source,'ok':ok}),flush=True)
            if not ok:failed.append(source)
    atomic_json(base/'data/prepared-v2/CONVERSION_STATUS.json',{'status':'failed' if failed else 'complete','failed':failed,'sources':sources})
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
