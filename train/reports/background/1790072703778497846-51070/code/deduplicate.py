"""Exact document deduplication with SHA256 audit report."""
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); seen=set(); total=kept=0
 with open(a.output,'w',encoding='utf-8') as out:
  for line in open(a.input,encoding='utf-8'):
   total+=1
   try:r=json.loads(line); text=r.get('text','')
   except Exception:continue
   h=hashlib.sha256(text.encode()).hexdigest()
   if h in seen: continue
   seen.add(h); out.write(json.dumps({'text':text},ensure_ascii=False)+'\n'); kept+=1
 Path(str(a.output)+'.report.json').write_text(json.dumps({'input':total,'kept':kept,'duplicates':total-kept},indent=2),encoding='utf-8')
if __name__=='__main__':main()
