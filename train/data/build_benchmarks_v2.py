"""Extract real held-out benchmark text, with per-dataset coverage reports."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def texts(name,row):
    if name=='mmlu': return [row['question'],*row['choices']]
    if name=='arc': return [row['question'],*row['choices']['text']]
    if name=='hellaswag': return [row['ctx'],*row['endings']]
    if name=='winogrande':return [row['sentence'],row['option1'],row['option2']]
    if name=='piqa':return [row['goal'],row['sol1'],row['sol2']]
    if name=='gsm8k':return [row['question'],row['answer']]
    if name=='humaneval':return [row['prompt'],row['canonical_solution']]
    raise ValueError(name)


def build(root,output):
    root=Path(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    specs={
      'mmlu':('benchmarks-mmlu',['all/test*.parquet']),
      'arc':('benchmarks-arc',['ARC-Challenge/test*.parquet','ARC-Easy/test*.parquet']),
      'hellaswag':('benchmarks-hellaswag',['data/validation*.parquet']),
      'winogrande':('benchmarks-winogrande',['winogrande_xl/validation*.parquet']),
      'piqa':('benchmarks-piqa',['plain_text/piqa-validation.parquet']),
      'gsm8k':('benchmarks-gsm8k',['main/test*.parquet']),
      'humaneval':('benchmarks-humaneval',['openai_humaneval/test*.parquet']),
    }
    reports={};total=0;out=output.with_name(output.name+'.incomplete')
    with out.open('w',encoding='utf-8') as f:
        for name,(directory,patterns) in specs.items():
            files=sorted({p for pattern in patterns for p in (root/directory).glob(pattern)})
            if name=='mmlu' and not files:files=sorted((root/directory).glob('*/test*.parquet'))
            if not files:raise FileNotFoundError(f'{name}: held-out parquet missing')
            count=eligible=0;seen=set();fileinfo=[]
            for path in files:
                h=hashlib.sha256(path.read_bytes()).hexdigest();fileinfo.append({'path':str(path),'sha256':h})
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        fields=texts(name,row)
                        if not all(isinstance(x,str) for x in fields):raise ValueError('Invalid benchmark field')
                        joined='\n'.join(fields)
                        key=hashlib.sha256(joined.encode()).hexdigest()
                        if key in seen:continue
                        seen.add(key);count+=1
                        # Candidate options are separate text pieces; do not join unrelated options into ngrams.
                        if any(len(x.split())>=13 for x in fields):eligible+=1
                        f.write(json.dumps({'benchmark':name,'id':key,'segments':fields,'text':joined},ensure_ascii=False)+'\n')
                        total+=1
            if not count or not eligible:raise ValueError(f'{name}: empty 13-gram coverage')
            reports[name]={'rows':count,'rows_with_13word_segment':eligible,'files':fileinfo}
    out.replace(output)
    report={'status':'complete','records':total,'datasets':reports,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'limits':['Only declared held-out splits covered; HumanEval test harness excluded','Coverage below 13 words reported, not silently claimed complete']}
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,ensure_ascii=False))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data/mini-k3/data/raw');p.add_argument('--output',required=True)
    a=p.parse_args();build(a.root,a.output)
