"""Schema-preserving raw conversion. Metadata files are never training rows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

SPECS = {
    'fineweb-edu': ('fineweb-edu','HuggingFaceFW/fineweb-edu','pretrain',['data/**/*.parquet']),
    'chinese-fineweb-edu': ('culturax','opencsg/chinese-fineweb-edu','pretrain',['IndustryCorpus/*.parquet','Skypile/*.parquet']),
    'cosmopedia': ('cosmopedia','AI-ModelScope/chinese-cosmopedia','pretrain',['data/*.parquet']),
    'finemath': ('finemath','HuggingFaceTB/finemath','pretrain',['finemath-3plus/train*.parquet']),
    'open-web-math': ('open-web-math','open-web-math/open-web-math','pretrain',['data/train*.parquet']),
    'dolma-body': ('dolma-body','modelscope/dolma','pretrain',['data/v1_7/*.json.gz']),
    'code-python': ('code-python','HuggingFaceCode/stack-v3-train','code',['data/*.parquet']),
    'openassistant': ('openassistant','OpenAssistant/oasst1','sft',['data/train*.parquet','data/validation*.parquet']),
    'openhermes': ('openhermes','teknium/OpenHermes-2.5','sft',['openhermes2_5.json']),
    'openr1': ('openr1','open-r1/OpenR1-Math-220k','sft',['all/default-*.parquet']),
    'ultrafeedback': ('ultrafeedback','argilla/ultrafeedback-binarized-preferences-cleaned','preference',['data/train*.parquet']),
}


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def json_array(stream):
    """Incremental standard JSON array reader; rejects malformed/trailing data."""
    decoder=json.JSONDecoder();buffer='';eof=False
    def refill():
        nonlocal buffer,eof
        data=stream.read(1024*1024)
        if not data: eof=True
        buffer+=data
    refill()
    while not buffer.lstrip() and not eof: refill()
    buffer=buffer.lstrip()
    if not buffer.startswith('['): raise ValueError('Expected a JSON array')
    buffer=buffer[1:];first=True
    while True:
        buffer=buffer.lstrip()
        if not buffer and not eof: refill();continue
        if buffer.startswith(']'):
            buffer=buffer[1:]
            while not eof: refill()
            if buffer.strip(): raise ValueError('Trailing JSON data')
            return
        if not first:
            if not buffer.startswith(','): raise ValueError('Expected array separator')
            buffer=buffer[1:].lstrip()
        while True:
            if not buffer and not eof: refill();buffer=buffer.lstrip()
            try: row,end=decoder.raw_decode(buffer);break
            except json.JSONDecodeError:
                if eof: raise
                refill()
        yield row
        buffer=buffer[end:];first=False


def rows(path):
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from b.to_pylist()
    else:
        op=gzip.open if path.name.endswith('.gz') else open
        with op(path,'rt',encoding='utf-8',errors='strict') as f:
            if path.suffix=='.json':
                yield from json_array(f)
            else:
                for number,line in enumerate(f,1):
                    if line.strip():
                        try: yield json.loads(line)
                        except json.JSONDecodeError as exc: raise ValueError(f'{path}:{number}: invalid JSON') from exc


def chat_messages(items):
    aliases={'human':'user','gpt':'assistant','prompter':'user'}
    result=[]
    for item in items:
        role=aliases.get(item.get('role',item.get('from')),item.get('role',item.get('from')))
        text=item.get('content',item.get('value'))
        if role not in ('system','user','assistant','tool') or not isinstance(text,str):
            raise ValueError('Invalid conversation turn')
        result.append({'role':role,'content':text})
    return result


def adapt(row, source, origin, counters):
    if not isinstance(row,dict): raise ValueError('Dataset row is not an object')
    role=SPECS[source][2]
    base={'source':source,'repo':SPECS[source][1],'origin':origin,'kind':role}
    if role=='code':
        for index,f in enumerate(row.get('files',[])):
            if not isinstance(f.get('content'),str): counters['code_missing_content']+=1;continue
            if str(f.get('language','')).lower()!='python': counters['non_python']+=1;continue
            if f.get('is_vendor'): counters['vendor']+=1;continue
            yield dict(base,kind='code',text=f['content'],repo_path=row.get('repo_path'),
                file_path=f.get('file_path'),language=f.get('language'),license_type=f.get('license_type'),
                detected_licenses=f.get('detected_licenses'),content_id=f.get('content_id'),file_index=index)
        return
    if role=='pretrain':
        text=next((row[k] for k in ('text','content','body') if isinstance(row.get(k),str)),None)
        if text is None: raise ValueError(f'{source}: missing text field')
        # Preserve mathematical/code whitespace and source fields; do not collapse lines.
        kind='code' if source=='dolma-body' and 'algebraic-stack' in origin['file'] else 'text'
        yield dict(base,kind=kind,text=text,upstream_id=row.get('id'),url=row.get('url'),metadata=row.get('metadata'))
    elif source=='openhermes':
        yield dict(base,messages=chat_messages(row['conversations']))
    elif source=='openr1':
        messages=chat_messages(row['messages'])
        yield dict(base,messages=messages,problem=row.get('problem'),answer=row.get('answer'),uuid=row.get('uuid'))
    elif role=='preference':
        chosen,rejected=chat_messages(row['chosen']),chat_messages(row['rejected'])
        if not chosen or not rejected or chosen[:-1]!=rejected[:-1]:
            raise ValueError('Preference pair has different prompt turns')
        if chosen[-1]['role']!='assistant' or rejected[-1]['role']!='assistant':
            raise ValueError('Preference completion must be assistant')
        yield dict(base,prompt=chosen[:-1],chosen=chosen[-1]['content'],rejected=rejected[-1]['content'],
            chosen_rating=row.get('chosen-rating'),rejected_rating=row.get('rejected-rating'))


def convert(source, root, output):
    directory,repo,role,patterns=SPECS[source]
    src=Path(root)/directory;dest=Path(output)/source;dest.mkdir(parents=True,exist_ok=True)
    files=sorted({p for glob in patterns for p in src.glob(glob) if p.is_file()})
    if not files: raise ValueError(f'{source}: no selected data files')
    if (dest/'COMPLETE.json').exists(): raise ValueError('Output already completed; verify/reuse or choose a new run directory')
    counters=Counter();origins=[];written=[];handle=None;pending=None;part=0;bytes_written=0;part_docs=0
    def close_part():
        nonlocal handle,pending
        if handle:
            handle.flush();os.fsync(handle.fileno());handle.close()
            final=dest/f'part-{part:05d}.jsonl';os.replace(pending,final)
            written.append({'path':str(final),'documents':part_docs,'bytes':final.stat().st_size,'sha256':digest_file(final)})
            handle=None
    def emit(record):
        nonlocal handle,pending,part,bytes_written,part_docs
        record['id']=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        encoded=(json.dumps(record,ensure_ascii=False)+'\n').encode()
        if handle is not None and bytes_written+len(encoded)>128*1024*1024:
            close_part();part+=1
        if handle is None:
            pending=dest/f'part-{part:05d}.jsonl.incomplete';handle=pending.open('wb');bytes_written=0;part_docs=0
        handle.write(encoded);bytes_written+=len(encoded);part_docs+=1;counters['output_records']+=1
    try:
        for path in files:
            origin={'file':str(path.relative_to(src)),'sha256':digest_file(path)}
            origins.append(dict(origin,bytes=path.stat().st_size))
            if source=='openassistant':
                data={}
                for n,row in enumerate(rows(path)):
                    counters['input_rows']+=1;data[row['message_id']]=(n,row)
                for mid,(n,row) in data.items():
                    if row.get('role')!='assistant' or row.get('deleted') or row.get('review_result') is False:continue
                    chain=[];visited=set();current=mid
                    while current is not None:
                        if current in visited or current not in data:chain=[];break
                        visited.add(current);_,node=data[current]
                        if node.get('deleted') or node.get('review_result') is False:chain=[];break
                        chain.append({'role':{'prompter':'user'}.get(node['role'],node['role']),'content':node['text']})
                        current=node.get('parent_id')
                    if chain:
                        emit({'source':source,'repo':repo,'origin':dict(origin,row=n),'kind':'sft',
                              'messages':list(reversed(chain)),'group_id':row.get('message_tree_id'),
                              'official_split':'validation' if 'validation' in path.name else 'train'})
                    else:counters['invalid_tree']+=1
            else:
                for n,row in enumerate(rows(path)):
                    counters['input_rows']+=1
                    for record in adapt(row,source,dict(origin,row=n),counters):emit(record)
            print(json.dumps({'source':source,'file':origin['file'],'counts':dict(counters)},ensure_ascii=False),flush=True)
        close_part()
        if not counters['output_records']:raise ValueError('No usable records produced')
        report={'source':source,'repo':repo,'role':role,'status':'complete','schema_version':2,
                'script_sha256':digest_file(__file__),'inputs':origins,'parts':written,'counts':dict(counters)}
        atomic_json(dest/'COMPLETE.json',report)
        return report
    finally:
        if handle:handle.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=SPECS,required=True)
    p.add_argument('--root',default='/data/mini-k3/data/raw');p.add_argument('--output',default='/data/mini-k3/data/prepared-v2/canonical')
    a=p.parse_args();convert(a.source,a.root,a.output)
