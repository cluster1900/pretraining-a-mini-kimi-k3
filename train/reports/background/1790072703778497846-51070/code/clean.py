"""Deterministic JSONL/JSONL.GZ cleaner with auditable rejection reasons."""
import argparse, gzip, hashlib, json, re
from pathlib import Path
EMAIL=re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"); SECRET=re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+")
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--min_chars',type=int,default=40); a=p.parse_args(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); stats={'input':0,'kept':0,'rejected':{}}
 op=gzip.open if str(a.output).endswith('.gz') else open
 with op(a.output,'wt',encoding='utf-8') as out:
  for line in open(a.input,encoding='utf-8',errors='replace'):
   stats['input']+=1
   try: r=json.loads(line); text=r.get('text') or r.get('content') or r.get('body') or ''
   except Exception: stats['rejected']['invalid_json']=stats['rejected'].get('invalid_json',0)+1; continue
   text=' '.join(str(text).replace('\x00',' ').split())
   reason=''
   if len(text)<a.min_chars: reason='too_short'
   elif SECRET.search(text): reason='secret'
   elif text.count('�')>2: reason='replacement_chars'
   if reason: stats['rejected'][reason]=stats['rejected'].get(reason,0)+1; continue
   out.write(json.dumps({'text':text},ensure_ascii=False)+'\n'); stats['kept']+=1
 Path(str(a.output)+'.report.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
if __name__=='__main__': main()
