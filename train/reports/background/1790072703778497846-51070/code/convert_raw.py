"""Convert downloaded parquet/JSON(.gz) files to canonical JSONL on /data."""
import argparse, gzip, json
from pathlib import Path
def rows(path):
 if path.suffix=='.parquet':
  import pyarrow.parquet as pq
  for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
   for r in batch.to_pylist(): yield r
 else:
  op=gzip.open if path.name.endswith('.gz') else open
  with op(path,'rt',encoding='utf-8',errors='replace') as f:
   raw=f.read()
  try:
   parsed=json.loads(raw)
   if isinstance(parsed,list):
    yield from parsed
   elif isinstance(parsed,dict): yield parsed
   return
  except Exception: pass
  for line in raw.splitlines():
   try: yield json.loads(line)
   except Exception: continue
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--input_root',default='/data/mini-k3/data/raw'); p.add_argument('--output_root',default='/data/mini-k3/data/converted'); a=p.parse_args()
 src=Path(a.input_root)/a.source; out=Path(a.output_root)/(a.source+'.jsonl'); out.parent.mkdir(parents=True,exist_ok=True); n=0
 with out.open('w',encoding='utf-8') as w:
  for path in sorted(src.rglob('*')):
   if path.suffix not in ('.parquet','.json','.jsonl') and not path.name.endswith(('.json.gz','.jsonl.gz')): continue
   for r in rows(path):
    if not isinstance(r,dict): continue
    text=r.get('text') or r.get('content') or r.get('body') or r.get('problem') or r.get('solution') or r.get('question') or r.get('prompt') or ''
    # Stack files store code in nested ``files`` records.
    if not text and isinstance(r.get('files'),list):
     text='\n'.join(str(x.get('content','')) for x in r['files'] if isinstance(x,dict) and x.get('content'))
    # OpenHermes records store chat turns in ``conversations``.
    if not text and isinstance(r.get('conversations'),list):
     text='\n'.join(f"{x.get('from','')}: {x.get('value','')}" for x in r['conversations'] if isinstance(x,dict))
    if not text and (r.get('prompt') or r.get('chosen') or r.get('rejected')):
     text=json.dumps({k:r.get(k) for k in ('prompt','chosen','rejected') if r.get(k) is not None},ensure_ascii=False)
    if isinstance(text,str) and text.strip(): w.write(json.dumps({'text':text,'source':a.source},ensure_ascii=False)+'\n'); n+=1
 print(json.dumps({'source':a.source,'documents':n,'output':str(out)}))
if __name__=='__main__':main()
