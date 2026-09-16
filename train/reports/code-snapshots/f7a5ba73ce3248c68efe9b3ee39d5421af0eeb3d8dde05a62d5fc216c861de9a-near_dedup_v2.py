"""Disk-backed MinHash-LSH near deduplication, with exact feature-set scoring.

Uses all normalized 5-unit shingles, 64 permutations, 8 bands and Jaccard>=0.9.
LSH is approximate candidate retrieval; reports never claim exhaustive recall.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib
import numpy as np
from canonical_v2 import atomic_json,digest_file
from contamination_v2 import units
from dedup_v2 import SOURCES,verify_part

MASK=(1<<64)-1
PRIME=np.uint64((1<<61)-1)
RNG=np.random.RandomState(42)
A=RNG.randint(1,int(PRIME),size=64,dtype=np.uint64)
B=RNG.randint(0,int(PRIME),size=64,dtype=np.uint64)


def features(text,code=False):
    ts=units(text,'code' if code else 'prose')
    if len(ts)<5:return np.array([],dtype='<u8')
    values=[zlib.crc32(t.encode())+1 for t in ts];h=0;power=pow(1_000_003,4,1<<64);result=[]
    for v in values[:5]:h=(h*1_000_003+v)&MASK
    result.append(h)
    for i in range(5,len(values)):
        h=(((h-values[i-5]*power)&MASK)*1_000_003+values[i])&MASK;result.append(h)
    return np.unique(np.asarray(result,dtype='<u8'))


def bands(values):
    sig=np.full(64,0xffffffff,dtype=np.uint64)
    with np.errstate(over='ignore'):
        for i in range(0,len(values),2048):
            hashes=((values[i:i+2048,None]*A+B)%PRIME)&np.uint64(0xffffffff)
            sig=np.minimum(sig,hashes.min(axis=0))
    return [bytes([i])+sig[i*8:(i+1)*8].astype('<u4').tobytes() for i in range(8)]


def similarity(a,b):
    intersection=len(np.intersect1d(a,b,assume_unique=True))
    return intersection/(len(a)+len(b)-intersection) if len(a)+len(b)>intersection else 1.0


def near_deduplicate(root):
    root=Path(root)
    sources={s:json.loads((root/'deduped'/s/'COMPLETE.json').read_text()) for s in SOURCES}
    identity=hashlib.sha256(json.dumps({s:digest_file(root/'deduped'/s/'COMPLETE.json') for s in SOURCES},sort_keys=True).encode()).hexdigest()
    with (root/'near-dedup.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        con=sqlite3.connect(root/'near-dedup.sqlite')
        con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=FULL')
        con.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
        con.execute('CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, features BLOB, n INTEGER)')
        con.execute('CREATE TABLE IF NOT EXISTS bands (key BLOB, doc INTEGER)');con.execute('CREATE INDEX IF NOT EXISTS band_key ON bands(key)')
        con.execute('CREATE TABLE IF NOT EXISTS parts (source TEXT, ordinal INTEGER, report TEXT, PRIMARY KEY(source,ordinal))')
        old=con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if old and old[0]!=identity:raise ValueError('Near-dedup inputs changed')
        con.execute("INSERT OR IGNORE INTO meta VALUES ('identity',?)",(identity,));con.commit()
        total=Counter()
        try:
            for source in SOURCES:
                outdir=root/'near-deduped'/source;outdir.mkdir(parents=True,exist_ok=True);counts=Counter();parts=[]
                for ordinal,part in enumerate(sources[source]['parts']):
                    old=con.execute('SELECT report FROM parts WHERE source=? AND ordinal=?',(source,ordinal)).fetchone()
                    if old:
                        report=json.loads(old[0]);verify_part(report)
                    else:
                        verify_part(part);out=outdir/f'part-{ordinal:05d}.jsonl';tmp=out.with_name(out.name+'.incomplete');c=Counter()
                        con.execute('BEGIN IMMEDIATE')
                        try:
                            with Path(part['path']).open(encoding='utf-8') as f,tmp.open('w',encoding='utf-8') as w:
                                for line in f:
                                    row=json.loads(line);c['input']+=1;duplicate=False
                                    if row['kind'] in ('text','code'):
                                        fs=features(row['text'],row['kind']=='code')
                                        if len(fs)>=20:
                                            keys=bands(fs)
                                            candidates=con.execute('SELECT DISTINCT d.id,d.features FROM bands b JOIN docs d ON d.id=b.doc WHERE b.key IN (?,?,?,?,?,?,?,?) AND d.n BETWEEN ? AND ? ORDER BY d.id',(*keys,int(.9*len(fs)),int(len(fs)/.9)+1))
                                            for _,blob in candidates:
                                                previous=np.frombuffer(zlib.decompress(blob),dtype='<u8')
                                                if similarity(fs,previous)>=.9:duplicate=True;break
                                            if not duplicate:
                                                did=con.execute('INSERT INTO docs(features,n) VALUES (?,?)',(zlib.compress(fs.tobytes(),1),len(fs))).lastrowid
                                                con.executemany('INSERT INTO bands VALUES (?,?)',[(key,did) for key in keys])
                                    if duplicate:c['near_duplicates']+=1;continue
                                    w.write(line);c['kept']+=1
                                w.flush();os.fsync(w.fileno())
                            os.replace(tmp,out);report={'path':str(out),'bytes':out.stat().st_size,'documents':c['kept'],'sha256':digest_file(out),'counts':dict(c)}
                            con.execute('INSERT INTO parts VALUES (?,?,?)',(source,ordinal,json.dumps(report)));con.commit()
                        except BaseException:con.rollback();raise
                    counts.update(report['counts']);parts.append(report)
                    atomic_json(outdir/'PROGRESS.json',{'source':source,'counts':dict(counts),'parts':len(parts)})
                if counts['input']!=counts['kept']+counts['near_duplicates']:raise ValueError('Near-dedup counters mismatch')
                atomic_json(outdir/'COMPLETE.json',{'status':'complete','source':source,'parts':parts,'counts':dict(counts),
                    'upstream_report_sha256':digest_file(root/'deduped'/source/'COMPLETE.json'),'script_sha256':digest_file(__file__),
                    'method':'5-unit shingles, 64 MinHash permutations, 8x8 LSH, feature Jaccard>=0.9; SFT/preference exact dedup only'})
                total.update(counts);print(json.dumps({'source':source,'counts':dict(counts)}),flush=True)
            atomic_json(root/'NEAR_DEDUP_COMPLETE.json',{'status':'complete','counts':dict(total),'input_identity':identity,'threshold':.9,
                'limits':['Approximate candidate recall','64-bit shingle fingerprints','Near dedup applies to pretraining documents, not alternate SFT answers']})
        finally:con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();near_deduplicate(a.root)
