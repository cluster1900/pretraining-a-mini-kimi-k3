"""Conservative cleaning preserving code whitespace, chat roles and provenance."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from canonical_v2 import atomic_json, digest_file

EMAIL=re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)')
SECRET=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b')
ALLOWED={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','unlicense','cc0-1.0','0bsd'}


def license_ids(record):
    result=set()
    for value in [record.get('license_type'),*(record.get('detected_licenses') or [])]:
        if isinstance(value,str):result.add(value.lower())
        elif isinstance(value,dict):
            for key in ('license','spdx_id','name'):
                if isinstance(value.get(key),str):result.add(value[key].lower())
    return result


def clean_text(text, is_code=False):
    if not isinstance(text,str): raise TypeError('Expected text')
    if SECRET.search(text):return None,'secret_pattern'
    text=unicodedata.normalize('NFC',text).replace('\r\n','\n').replace('\r','\n')
    if '\x00' in text:return None,'nul_byte'
    if text.count('\ufffd')>max(2,len(text)//100):return None,'encoding_damage'
    # No split/join or dedent: Python indentation, math layout and line breaks matter.
    if not is_code:text=EMAIL.sub('[EMAIL]',text)
    return text,None


def clean_record(record):
    r=copy.deepcopy(record);kind=r['kind'];is_code=kind=='code'
    if r['source']=='code-python' and not (license_ids(r)&ALLOWED):return None,'code_license_not_allowlisted'
    if kind in ('text','code'):
        text,reason=clean_text(r['text'],is_code)
        if reason:return None,reason
        if len(text.strip()) < (20 if is_code else 40):return None,'too_short'
        lines=[x for x in text.splitlines() if x.strip()]
        if not is_code and len(lines)>=20 and len(set(lines))/len(lines)<0.1:return None,'repeated_lines'
        r['text']=text
    elif kind=='sft':
        messages=r['messages']
        if not messages or messages[-1]['role']!='assistant':return None,'no_final_assistant'
        if not any(x['role']=='user' for x in messages):return None,'no_user_prompt'
        previous=None
        for turn in messages:
            text,reason=clean_text(turn['content'],True)
            if reason:return None,reason
            if not text.strip():return None,'empty_turn'
            if previous==turn['role'] and previous in ('user','assistant'):return None,'repeated_role'
            turn['content']=text;previous=turn['role']
    elif kind=='preference':
        if not r['prompt'] or r['prompt'][-1]['role']!='user':return None,'invalid_preference_prompt'
        for key in ('chosen','rejected'):
            text,reason=clean_text(r[key],True)
            if reason:return None,reason
            if not text.strip():return None,'empty_completion'
            r[key]=text
        if r['chosen']==r['rejected']:return None,'identical_completions'
    else:raise ValueError('Unknown canonical kind: '+kind)
    payload={key:r[key] for key in ('text','messages','prompt','chosen','rejected') if key in r}
    r['content_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return r,None


def clean_source(source,root):
    base=Path(root);src=base/'canonical'/source;dest=base/'cleaned'/source
    upstream=json.loads((src/'COMPLETE.json').read_text());dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():raise ValueError('Completed output exists; validate it explicitly')
    stats=Counter();parts=[]
    for item in upstream['parts']:
        path=Path(item['path'])
        if digest_file(path)!=item['sha256']:raise ValueError('Canonical hash mismatch')
        out=dest/path.name;tmp=out.with_name(out.name+'.incomplete');n=0
        with path.open(encoding='utf-8') as f,tmp.open('w',encoding='utf-8') as w:
            for line in f:
                stats['input']+=1
                row=json.loads(line);r,reason=clean_record(row)
                if reason:stats['reject_'+reason]+=1;continue
                w.write(json.dumps(r,ensure_ascii=False)+'\n');n+=1;stats['kept']+=1
            w.flush();os.fsync(w.fileno())
        os.replace(tmp,out)
        parts.append({'path':str(out),'documents':n,'bytes':out.stat().st_size,'sha256':digest_file(out)})
        atomic_json(dest/'PROGRESS.json',{'source':source,'counts':dict(stats),'completed_parts':len(parts)})
    if not stats['kept']:raise ValueError(f'{source}: all records rejected')
    assert stats['input']==stats['kept']+sum(v for k,v in stats.items() if k.startswith('reject_'))
    atomic_json(dest/'COMPLETE.json',{'source':source,'status':'complete','parts':parts,'counts':dict(stats),
        'script_sha256':digest_file(__file__),'upstream_report_sha256':digest_file(src/'COMPLETE.json'),
        'limits':['PII filtering covers email in prose and explicit secret-key patterns; not a guarantee of total PII removal']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',default='/data/mini-k3/data/prepared-v2')
    a=p.parse_args();clean_source(a.source,a.root)
