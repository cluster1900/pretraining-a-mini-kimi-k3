"""Exact 13-unit matching with rolling hashes; hash hits get exact verification."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import zlib
from canonical_v2 import atomic_json, digest_file

N=13
BASE=1_000_003
MASK=(1<<64)-1
POWER=pow(BASE,N-1,1<<64)
WORD=re.compile(r'[\u3400-\u9fff]|[A-Za-z0-9]+|[^\W_]+',re.UNICODE)
CODE=re.compile(r'\w+|[^\w\s]',re.UNICODE)


def units(text,mode):
    s=unicodedata.normalize('NFKC',text)
    return (CODE if mode=='code' else WORD).findall(s if mode=='code' else s.casefold())


def windows(terms):
    if len(terms)<N:return
    values=[zlib.crc32(t.encode('utf-8'))+1 for t in terms]
    h=0
    for value in values[:N]:h=(h*BASE+value)&MASK
    yield 0,h
    for end in range(N,len(values)):
        h=(((h-values[end-N]*POWER)&MASK)*BASE+values[end])&MASK
        yield end-N+1,h


def build_index(records):
    tables={'prose':{},'code':{}};counts=Counter();covered=Counter()
    for r in records:
        name=r['benchmark'];mode='code' if name=='humaneval' else 'prose';counts[name]+=1;usable=False
        for segment in r['segments']:
            terms=units(segment,mode)
            for i,h in windows(terms):
                window=tuple(terms[i:i+N]);bucket=tables[mode].setdefault(h,[])
                if window not in bucket:bucket.append(window)
                usable=True
        covered[name]+=int(usable)
    if not counts or any(not covered[k] for k in counts):
        raise ValueError('A required benchmark has zero indexable 13-grams')
    if not any(tables.values()):raise ValueError('Empty benchmark index is forbidden')
    return tables,{'rows':dict(counts),'covered_rows':dict(covered),
                   'ngrams':{m:sum(map(len,t.values())) for m,t in tables.items()},'n':N}


def match(text,tables):
    # Search both modes because code can appear inside prose/chat responses.
    for mode,table in tables.items():
        if not table:continue
        terms=units(text,mode)
        for i,h in windows(terms):
            if h in table and tuple(terms[i:i+N]) in table[h]:return True
    return False


def record_texts(row):
    if row['kind'] in ('code','text'):return [row['text']]
    if row['kind']=='sft':return [t['content'] for t in row['messages']]
    if row['kind']=='preference':return [t['content'] for t in row['prompt']]+[row['chosen'],row['rejected']]
    raise ValueError('Unsupported canonical record')


def compile_index(input_path,output):
    with Path(input_path).open(encoding='utf-8') as f:tables,report=build_index(json.loads(x) for x in f)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.incomplete')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(tables,f,ensure_ascii=False)
    os.replace(tmp,output)
    atomic_json(output.with_suffix('.report.json'),dict(report,status='complete',benchmark_sha256=digest_file(input_path),
        index_sha256=digest_file(output),script_sha256=digest_file(__file__),
        normalization='NFKC+casefold prose words/Han characters; case-sensitive code lexemes; exact hit verification'))
    print(json.dumps(report),flush=True)


def load_index(path):
    report=json.loads(Path(path).with_suffix('.report.json').read_text())
    if digest_file(path)!=report['index_sha256']:raise ValueError('Index hash mismatch')
    with gzip.open(path,'rt',encoding='utf-8') as f:raw=json.load(f)
    tables={m:{int(h):[tuple(w) for w in ws] for h,ws in t.items()} for m,t in raw.items()}
    if not any(tables.values()):raise ValueError('Empty index')
    return tables,report


def filter_source(source,root,index_path,input_stage='near-deduped'):
    root=Path(root);src=root/input_stage/source;outdir=root/'decontaminated'/source
    marker=src/'COMPLETE.json';up=json.loads(marker.read_text())
    tables,index_report=load_index(index_path)
    outdir.mkdir(parents=True,exist_ok=True);parts=[];counts=Counter()
    if (outdir/'COMPLETE.json').exists():raise ValueError('Output already completed')
    for item in up['parts']:
        path=Path(item['path'])
        if digest_file(path)!=item['sha256']:raise ValueError('Upstream data hash mismatch')
        out=outdir/path.name;tmp=out.with_name(out.name+'.incomplete');kept=0
        with path.open(encoding='utf-8') as r,tmp.open('w',encoding='utf-8') as w:
            for line in r:
                row=json.loads(line);counts['input']+=1
                if any(match(text,tables) for text in record_texts(row)):
                    counts['removed']+=1;continue
                w.write(line);counts['kept']+=1;kept+=1
            w.flush();os.fsync(w.fileno())
        os.replace(tmp,out);parts.append({'path':str(out),'documents':kept,'bytes':out.stat().st_size,'sha256':digest_file(out)})
        atomic_json(outdir/'PROGRESS.json',{'source':source,'counts':dict(counts),'parts':len(parts)})
    if counts['input']!=counts['removed']+counts['kept']:raise ValueError('Counters do not balance')
    if not counts['kept']:raise ValueError('No uncontaminated records')
    atomic_json(outdir/'COMPLETE.json',{'source':source,'status':'complete','counts':dict(counts),'parts':parts,
        'index_sha256':index_report['index_sha256'],'index_ngrams':index_report['ngrams'],
        'upstream_report_sha256':digest_file(marker),'script_sha256':digest_file(__file__)})

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build');b.add_argument('--input',required=True);b.add_argument('--output',required=True)
    f=sub.add_parser('filter');f.add_argument('--source',required=True);f.add_argument('--root',required=True);f.add_argument('--index',required=True);f.add_argument('--input-stage',default='near-deduped')
    a=p.parse_args()
    if a.mode=='build':compile_index(a.input,a.output)
    else:filter_source(a.source,a.root,a.index,a.input_stage)
