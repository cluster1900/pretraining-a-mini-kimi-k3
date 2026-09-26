"""Resume the data pipeline after raw conversion; never starts model training."""
import argparse, json, shutil, subprocess, time
from pathlib import Path
SOURCES=['fineweb-edu','culturax','cosmopedia','finemath','open-web-math','dolma-body','code-python','openassistant','openhermes','openr1','ultrafeedback']
PRETRAIN={'fineweb-edu':('fineweb-edu',.45),'culturax':('chinese-fineweb-edu',.15),'cosmopedia':('cosmopedia',.10),'finemath':('finemath',.05),'open-web-math':('open-web-math',.05),'dolma-body':('dolma-body',.10),'code-python':('code-python',.10)}
def run(cmd,log):
 with open(log,'ab') as f: return subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
def main():
 raise RuntimeError('Legacy pipeline invalidated by provenance/tokenizer/empty-benchmark audit; use schema-preserving v2 outputs only')
 p=argparse.ArgumentParser(); p.add_argument('--root',default='/data/mini-k3'); a=p.parse_args(); b=Path(a.root); conv=b/'data/converted'; clean=b/'data/cleaned'; dedup=b/'data/deduped'; decon=b/'data/decontaminated'; tok=b/'data/tokenized'; logs=b/'logs';
 while not all((clean/f'{s}.jsonl.report.json').exists() for s in SOURCES): time.sleep(60)
 jobs=[]
 for s in SOURCES:
  if not (dedup/f'{s}.jsonl.report.json').exists(): jobs.append(run([str(b/'venv/bin/python'),'-u',str(b/'project/train/data/deduplicate.py'),'--input',str(clean/f'{s}.jsonl'),'--output',str(dedup/f'{s}.jsonl')],logs/f'dedup-{s}.log'))
 for j in jobs:
  if j.wait()!=0: raise RuntimeError('deduplication failed; inspect dedup logs')
 bench=list(conv.glob('benchmarks-*.jsonl'))
 if not bench: raise RuntimeError('No benchmark JSONL available for decontamination')
 idx=b/'data/benchmark_index.jsonl'; idx.parent.mkdir(parents=True,exist_ok=True)
 with idx.open('w',encoding='utf-8') as w:
  for f in bench: w.write(f.read_text(encoding='utf-8',errors='replace'))
 jobs=[]
 for s in SOURCES:
  if not (decon/f'{s}.jsonl.report.json').exists(): jobs.append(run([str(b/'venv/bin/python'),'-u',str(b/'project/train/data/decontaminate.py'),'--input',str(dedup/f'{s}.jsonl'),'--benchmark',str(idx),'--output',str(decon/f'{s}.jsonl')],logs/f'decontam-{s}.log'))
 for j in jobs:
  if j.wait()!=0: raise RuntimeError('decontamination failed; inspect decontam logs')
 tok.mkdir(parents=True,exist_ok=True)
 for raw,(name,weight) in PRETRAIN.items():
  report=tok/name/f'{name}_shard_00000.bin'
  if report.exists(): continue
  job=run([str(b/'venv/bin/python'),'-u',str(b/'project/train/data/build_shards.py'),'--source_name',name,'--input_files',str(decon/f'{raw}.jsonl'),'--tokenizer_model',str(b/'data/tokenizer'),'--output_dir',str(tok),'--manifest_path',str(b/'data/manifests/pretrain_stable.json'),'--weight',str(weight)],logs/f'tokenize-{name}.log')
  if job.wait()!=0: raise RuntimeError('tokenization failed for '+name)
 print('pipeline stages complete; validate manifest and run smoke test explicitly')
if __name__=='__main__': main()
