"""Remove documents matching benchmark 13-grams before tokenization."""
import argparse,json,re
from pathlib import Path
def norm(s): return re.sub(r'\s+',' ',s.lower()).strip()
def grams(s,n=13):
 w=norm(s).split(); return {' '.join(w[i:i+n]) for i in range(max(0,len(w)-n+1))}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--benchmark',required=True); p.add_argument('--output',required=True); a=p.parse_args(); idx=set()
 for line in open(a.benchmark,encoding='utf-8'):
  try:r=json.loads(line); idx |= grams(r.get('text') or r.get('question') or r.get('prompt',''))
  except Exception:pass
 if not idx: raise ValueError('Benchmark index has zero 13-grams; refusing ineffective decontamination')
 total=removed=0; Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',encoding='utf-8') as out:
  for line in open(a.input,encoding='utf-8'):
   total+=1
   try:r=json.loads(line); gs=grams(r.get('text',''))
   except Exception:continue
   if gs & idx: removed+=1; continue
   out.write(json.dumps(r,ensure_ascii=False)+'\n')
 Path(str(a.output)+'.report.json').write_text(json.dumps({'benchmark_13grams':len(idx),'input':total,'removed':removed,'kept':total-removed},indent=2),encoding='utf-8')
if __name__=='__main__':main()
