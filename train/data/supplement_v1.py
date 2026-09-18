"""Download and process token-deficit supplements in an isolated run root.

The script waits for the current prepared-v2 run, selects explicit files from
the fixed FineWeb inventory and a pinned GitHub-Code dataset revision, records
every download and transformation report, then rebuilds all affected stages.
It never launches model training.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from canonical_v2 import atomic_json, digest_file

ALLOWED = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'unlicense', '0bsd'}


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


class Run:
    def __init__(self, work):
        self.work=Path(work); self.base=self.work/'data'; self.raw=self.base/'raw'
        self.root=self.base/'prepared-v2-supplement-v1'; self.reports=self.base/'reports/supplement-v1'
        self.logs=self.work/'logs/prepared-v2-supplement-v1'; self.state=self.reports/'PIPELINE_STATUS.json'
        self.reports.mkdir(parents=True,exist_ok=True); self.logs.mkdir(parents=True,exist_ok=True)
        self.status={'status':'running','stage':'init','started_at':now(),'root':str(self.root)}
        atomic_json(self.state,self.status)
    def update(self,stage,status=None,**kw):
        self.status.update(stage=stage,updated_at=now(),**kw)
        if status:self.status['status']=status
        atomic_json(self.state,self.status)
    def report(self,name,data):
        data=dict(data,generated_at=now())
        atomic_json(self.reports/name,data)
        self.update(self.status['stage'],report=str(self.reports/name),report_sha256=digest_file(self.reports/name))
    def run(self,cmd,logname):
        log=self.logs/logname
        with log.open('ab') as handle:
            child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            code=child.wait()
        if code: raise RuntimeError(f'command failed exit={code}: {log}')


def wait_base(run, poll=60):
    while True:
        state=run.base/'prepared-v2/PIPELINE_STATUS.json'
        data=json.loads(state.read_text()) if state.exists() else {}
        if data.get('status')=='complete': return data
        if data.get('status')=='failed': raise RuntimeError('prepared-v2 failed: '+str(data.get('error')))
        run.update('waiting_for_base',base_status=data.get('status'),base_stage=data.get('stage'))
        time.sleep(poll)


def select_fineweb(run, target_bytes=12_000_000_000):
    existing=list((run.raw/'fineweb-edu').glob('data/*/*.parquet'))
    if len(existing)>=8:
        return {'source':'HuggingFaceFW/fineweb-edu','files':[],'selected_bytes':0,'selection':'cached'}
    inv=json.loads((run.raw/'fineweb-edu/INVENTORY.json').read_text())
    groups={}
    for item in inv['files']:
        if not (item['path'].startswith('data/') and item['path'].endswith('.parquet')): continue
        path=item['path']; p=run.raw/'fineweb-edu'/path
        if p.is_file() and p.stat().st_size==item['size']: continue
        group=path.split('/')[1] if path.count('/')>=2 else path
        groups.setdefault(group,[]).append(item)
    selected=[]; total=0
    group_items=sorted(groups.items())
    index=0
    while total<target_bytes and group_items:
        group,items=group_items[index%len(group_items)]
        if items:
            item=items.pop(0); selected.append(item); total+=item['size']
        index+=1
        if all(not x[1] for x in group_items): break
    return {'source':'HuggingFaceFW/fineweb-edu','revision':inv['revision'],'target_bytes':target_bytes,
            'selected_bytes':total,'files':selected,'selection':'round-robin across crawl directories'}


def download_fineweb(run, selection):
    from modelscope.hub.file_download import dataset_file_download
    out=[]
    for item in selection['files']:
        dst=run.raw/'fineweb-edu'/item['path']; dst.parent.mkdir(parents=True,exist_ok=True)
        actual=dataset_file_download('HuggingFaceFW/fineweb-edu',item['path'],revision='master',local_dir=str(run.raw/'fineweb-edu'),cache_dir=str(run.work/'ms-cache'))
        p=Path(actual)
        if p.resolve()!=dst.resolve() or p.stat().st_size!=item['size']:
            raise ValueError('FineWeb selected file mismatch: '+item['path'])
        digest=digest_file(p)
        if item.get('sha256') and digest!=item['sha256']: raise ValueError('FineWeb checksum mismatch: '+item['path'])
        out.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest,'upstream_sha256':item.get('sha256')})
    return out


def codeparrot_selection(count=200):
    req=urllib.request.Request('https://hf-mirror.com/api/datasets/codeparrot/github-code',headers={'User-Agent':'Mozilla/5.0'})
    meta=json.load(urllib.request.urlopen(req,timeout=60))
    revision=meta['sha']; files=[x['rfilename'] for x in meta['siblings'] if x.get('rfilename','').endswith('.parquet')]
    if not files: raise ValueError('No CodeParrot parquet shards found')
    step=max(1,len(files)//count); chosen=files[::step][:count]
    return {'source':'codeparrot/github-code','revision':revision,'files':chosen,'selection':'evenly spaced parquet shards','dataset_license':meta.get('cardData',{}).get('license')}


def download_codeparrot(run, selection, max_retries=5):
    out=[]
    dstroot=run.raw/'github-code';dstroot.mkdir(parents=True,exist_ok=True)
    for path in selection['files']:
        dst=dstroot/Path(path).name; tmp=dst.with_name(dst.name+'.incomplete')
        if dst.is_file() and dst.stat().st_size > 0:
            digest=digest_file(dst)
            out.append({'path':str(dst),'source_path':path,'bytes':dst.stat().st_size,'sha256':digest})
            continue
        url=f"https://hf-mirror.com/datasets/codeparrot/github-code/resolve/{selection['revision']}/{path}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        for attempt in range(1, max_retries + 1):
            try:
                if tmp.exists():
                    tmp.unlink()
                with urllib.request.urlopen(req, timeout=120) as response, open(tmp, 'wb') as f:
                    while True:
                        chunk = response.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                if tmp.is_file() and tmp.stat().st_size > 0:
                    break
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Download failed for {url} after {max_retries} attempts: {e}")
                time.sleep(attempt * 2)
        digest=digest_file(tmp); os.replace(tmp,dst)
        out.append({'path':str(dst),'source_path':path,'bytes':dst.stat().st_size,'sha256':digest})
    return out


def convert_codeparrot(run, files):
    import pyarrow as pa, pyarrow.parquet as pq
    outdir=run.raw/'code-python/data';outdir.mkdir(parents=True,exist_ok=True); outputs=[]; kept=0; rejected={}
    for ordinal,item in enumerate(files):
        out=outdir/f'github-supplement-{ordinal:04d}.parquet'
        if out.is_file() and out.stat().st_size > 0:
            digest=digest_file(out)
            outputs.append({'path':str(out),'sha256':digest,'cached':True})
            continue
        rows=[]; table=pq.read_table(item['path'])
        for row in table.to_pylist():
            path=row.get('path','')
            if not path.endswith('.py'): rejected['non_python']=rejected.get('non_python',0)+1; continue
            lic=str(row.get('license','')).lower()
            code=row.get('content')
            if lic not in ALLOWED: rejected['license_not_allowlisted']=rejected.get('license_not_allowlisted',0)+1; continue
            if not isinstance(code,str) or len(code.strip())<20: rejected['too_short']=rejected.get('too_short',0)+1; continue
            rows.append({'repo_path':row.get('repo_name'),'files':[{'content':code,'language':'Python','file_path':path,'license_type':lic,'is_vendor':False}]})
        pq.write_table(pa.Table.from_pylist(rows),out)
        digest=digest_file(out);outputs.append({'path':str(out),'rows':len(rows),'bytes':out.stat().st_size,'sha256':digest});kept+=len(rows)
    return {'files':outputs,'kept_rows':kept,'rejected':rejected,'license_allowlist':sorted(ALLOWED)}


def clone_root(run):
    if run.root.exists(): raise ValueError('Supplement root already exists')
    subprocess.run(['cp','-al',str(run.base/'prepared-v2'),str(run.root)],check=True)
    for stage in ('canonical','cleaned'):
        shutil.move(str(run.root/stage/'fineweb-edu'),str(run.root/stage/'fineweb-edu-before-supplement'))
        shutil.move(str(run.root/stage/'code-python'),str(run.root/stage/'code-python-before-supplement'))
    for stage in ('deduped','near-deduped','decontaminated','tokenized'):
        shutil.move(str(run.root/stage),str(run.root/(stage+'-before-supplement')));(run.root/stage).mkdir()
    for p in run.root.glob('*sqlite'):
        p.rename(p.with_name(p.name+'-before-supplement'))
    for p in run.root.glob('*COMPLETE.json'):
        p.rename(p.with_name(p.name.replace('.json','-before-supplement.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',default='/data/mini-k3');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();run=Run(args.work)
    try:
        wait_base(run)
        run.update('selecting')
        fine=select_fineweb(run); code=codeparrot_selection()
        atomic_json(run.reports/'selection.json',{'fineweb':fine,'codeparrot':code})
        run.update('downloading',selection_report=str(run.reports/'selection.json'))
        fine_files=download_fineweb(run,fine); code_files=download_codeparrot(run,code)
        run.report('download.json',{'status':'complete','fineweb_files':fine_files,'codeparrot_files':code_files})
        run.update('adapting_code'); adapted=convert_codeparrot(run,code_files);run.report('code-adapter.json',dict(status='complete',**adapted))
        run.update('cloning_base');clone_root(run)
        scripts=run.work/'project/train/data';py=run.work/'venv/bin/python'
        for source in ('fineweb-edu','code-python'):
            run.update('canonical_'+source);run.run([str(py),'-u',str(scripts/'canonical_v2.py'),'--source',source,'--root',str(run.raw),'--output',str(run.root/'canonical')],f'canonical-{source}.log')
            run.update('cleaned_'+source);run.run([str(py),'-u',str(scripts/'clean_v2.py'),'--source',source,'--root',str(run.root)],f'cleaned-{source}.log')
        run.update('pipeline')
        run.run([str(py),'-u',str(scripts/'continue_v2.py'),'--root',str(run.root),'--work',str(run.work),'--workers',str(args.workers),'--log-dir',str(run.logs)],'controller.log')
        run.update('coverage');run.run([str(py),str(scripts/'assess_training_coverage.py'),'--root',str(run.root),'--report',str(run.reports/'coverage.json')],'coverage.log')
        coverage=json.loads((run.reports/'coverage.json').read_text());run.update('complete' if coverage['status']=='sufficient_fixed_mix' else 'needs_supplement',status='complete' if coverage['status']=='sufficient_fixed_mix' else 'needs_supplement',coverage=coverage)
    except Exception as exc:
        run.update(run.status.get('stage','failed'),status='failed',error=str(exc));raise


if __name__=='__main__':main()
