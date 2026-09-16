"""Fail-loud manifest and shard integrity audit."""
import argparse,hashlib,json,os
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); a=p.parse_args(); m=json.load(open(a.manifest)); total=0
 for name,s in m.get('sources',{}).items():
  if not s.get('shards'): raise SystemExit(f'{name}: no shards')
  if s.get('weight',0)<0: raise SystemExit(f'{name}: negative weight')
  for q in s['shards']:
   if not os.path.isfile(q): raise SystemExit(f'{name}: missing {q}')
   size=os.path.getsize(q)
   if size%4: raise SystemExit(f'{name}: non-uint32 shard {q}')
   total+=size//4
 if not total: raise SystemExit('manifest has zero tokens')
 print(json.dumps({'sources':len(m['sources']),'tokens':total,'status':'ok'}))
if __name__=='__main__':main()
