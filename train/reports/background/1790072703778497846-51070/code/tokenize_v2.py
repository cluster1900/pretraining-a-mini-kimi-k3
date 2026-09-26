"""Encode complete structured sources with document-disjoint validation data."""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from canonical_v2 import atomic_json, digest_file
from tokenizer import K3Tokenizer


class TokenWriter:
    def __init__(self,directory,limit=50_000_000):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        if limit<=0:raise ValueError('Invalid shard capacity')
        self.limit=limit;self.parts=[];self.count=0;self.eos=0;self.handle=None;self.hash=None;self.total=0
    def write(self,ids,eos_id):
        position=0
        while position<len(ids):
            if self.handle is None:
                self.temp=self.directory/f'shard-{len(self.parts):05d}.bin.incomplete'
                self.handle=self.temp.open('wb');self.hash=hashlib.sha256();self.count=self.eos=0
            take=min(len(ids)-position,self.limit-self.count)
            values=ids[position:position+take]
            data=array('I',values)
            if data.itemsize!=4:raise RuntimeError('uint32 requires 4-byte unsigned int')
            if sys.byteorder!='little':data.byteswap()
            raw=data.tobytes();self.handle.write(raw);self.hash.update(raw)
            self.count+=take;self.total+=take;self.eos+=values.count(eos_id);position+=take
            if self.count==self.limit:self.flush()
    def flush(self):
        if self.handle is None:return
        self.handle.flush();os.fsync(self.handle.fileno());self.handle.close();self.handle=None
        final=self.temp.with_suffix('');os.replace(self.temp,final)
        item={'path':str(final),'tokens':self.count,'bytes':4*self.count,'eos':self.eos,'sha256':self.hash.hexdigest()}
        self.parts.append(item);atomic_json(final.with_suffix('.json'),item)
    def close(self):self.flush()


def encode_messages(messages,tokenizer):
    ids=[];labels=[]
    for turn in messages:
        header=tokenizer.encode(f"<|{turn['role']}|>\n",append_eos=False)
        body=tokenizer.encode(turn['content']+'\n',append_eos=turn['role']=='assistant')
        ids.extend(header);labels.extend([-100]*len(header));ids.extend(body)
        labels.extend(body if turn['role']=='assistant' else [-100]*len(body))
    if not any(x!=-100 for x in labels[1:]):raise ValueError('SFT record contains no supervised target')
    return ids,labels


def encode_preference(row,tokenizer):
    prompt=[]
    for turn in row['prompt']:
        prompt+=tokenizer.encode(f"<|{turn['role']}|>\n"+turn['content']+'\n',append_eos=False)
    prompt+=tokenizer.encode('<|assistant|>\n',append_eos=False)
    chosen=prompt+tokenizer.encode(row['chosen'],append_eos=True)
    rejected=prompt+tokenizer.encode(row['rejected'],append_eos=True)
    return {'prompt_len':len(prompt),'prompt_ids':prompt,'chosen_ids':chosen,'rejected_ids':rejected}


def encode_source(source,root,tokenizer_dir,shard_tokens=50_000_000):
    root=Path(root);src=root/'decontaminated'/source;marker=src/'COMPLETE.json'
    up=json.loads(marker.read_text());tok=K3Tokenizer(tokenizer_dir)
    out=root/'tokenized'/source;out.mkdir(parents=True,exist_ok=True)
    complete=out/'COMPLETE.json'
    if complete.exists():
        report=json.loads(complete.read_text())
        if report['tokenizer_fingerprint']!=tok.fingerprint or report['upstream_sha256']!=digest_file(marker):
            raise ValueError('Existing tokenization has different inputs/tokenizer')
        for item in report['files']:
            p=Path(item['path'])
            if p.stat().st_size!=item['bytes'] or digest_file(p)!=item['sha256']:raise ValueError('Existing token file corrupted')
        return report
    if list(out.glob('**/*.bin')) or list(out.glob('**/*.incomplete')) or list(out.glob('**/*.jsonl')):
        raise ValueError('Incomplete tokenization attempt exists; preserve it and choose a clean source output')
    kind='preference' if source=='ultrafeedback' else 'sft' if source in ('openassistant','openhermes','openr1') else 'pretrain'
    writers={s:TokenWriter(out/s,shard_tokens) for s in ('train','validation')} if kind=='pretrain' else {}
    text_outputs={s:(out/f'{s}.jsonl.incomplete').open('w',encoding='utf-8') for s in ('train','validation')} if kind!='pretrain' else {}
    index=out/'documents.jsonl.incomplete';counts=Counter();files=[]
    try:
        with index.open('w',encoding='utf-8') as ledger:
            for part in up['parts']:
                p=Path(part['path'])
                if p.stat().st_size!=part['bytes'] or digest_file(p)!=part['sha256']:raise ValueError('Decontaminated input hash mismatch')
                for line in p.open(encoding='utf-8'):
                    row=json.loads(line);split=row['split']
                    if split not in ('train','validation'):raise ValueError('Missing/invalid document split')
                    counts['documents']+=1;counts[split+'_documents']+=1
                    entry={'id':row['id'],'source':source,'split':split,'split_group':row['split_group'],'content_sha256':row['content_sha256']}
                    if kind=='pretrain':
                        ids=tok.encode(row['text'])
                        if any(t<0 or t>=tok.vocab_size for t in ids) or ids.count(tok.eos_token_id)!=1:raise ValueError('Invalid token ID or EOS injection')
                        entry.update(token_start=writers[split].total,tokens=len(ids))
                        writers[split].write(ids,tok.eos_token_id)
                        counts[split+'_tokens']+=len(ids)
                    elif kind=='sft':
                        ids,labels=encode_messages(row['messages'],tok)
                        record=dict(entry,input_ids=ids,labels=labels)
                        counts[split+'_tokens']+=len(ids);counts['supervised_tokens']+=sum(x!=-100 for x in labels)
                        text_outputs[split].write(json.dumps(record,ensure_ascii=False)+'\n')
                    else:
                        record=dict(entry,**encode_preference(row,tok))
                        counts[split+'_tokens']+=len(record['chosen_ids'])+len(record['rejected_ids'])
                        text_outputs[split].write(json.dumps(record,ensure_ascii=False)+'\n')
                    ledger.write(json.dumps(entry,ensure_ascii=False)+'\n')
                atomic_json(out/'PROGRESS.json',{'source':source,'counts':dict(counts),'input_part':part['path']})
            ledger.flush();os.fsync(ledger.fileno())
        if not counts['documents']:raise ValueError('Empty tokenization source')
        for split,writer in writers.items():
            writer.close();files.extend(dict(item,split=split) for item in writer.parts)
            if sum(item['eos'] for item in writer.parts)!=counts[split+'_documents']:raise ValueError('EOS/document count mismatch')
        for split,handle in text_outputs.items():
            handle.flush();os.fsync(handle.fileno());handle.close()
            final=out/f'{split}.jsonl';os.replace(out/f'{split}.jsonl.incomplete',final)
            files.append({'path':str(final),'split':split,'bytes':final.stat().st_size,'sha256':digest_file(final)})
        final_index=out/'documents.jsonl';os.replace(index,final_index)
        report={'status':'complete','source':source,'kind':kind,'counts':dict(counts),'files':files,
            'tokenizer_fingerprint':tok.fingerprint,'tokenizer_assets':tok.asset_hashes,'vocab_size':tok.vocab_size,
            'dtype':'<u4' if kind=='pretrain' else 'jsonl','upstream_sha256':digest_file(marker),
            'script_sha256':digest_file(__file__),'document_index':{'path':str(final_index),'sha256':digest_file(final_index)}}
        atomic_json(complete,report)
        print(json.dumps({'source':source,'counts':dict(counts)}),flush=True)
        return report
    finally:
        for writer in writers.values():
            if writer.handle:writer.handle.close()
        for handle in text_outputs.values():
            if not handle.closed:handle.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--shard-tokens',type=int,default=50_000_000)
    a=p.parse_args();encode_source(a.source,a.root,a.tokenizer,a.shard_tokens)
