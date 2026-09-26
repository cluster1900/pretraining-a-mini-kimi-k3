"""Direct hf-mirror dataset downloader; bypasses datasets API routing issues."""
import argparse, hashlib, json, os, urllib.request, subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument("--repo",required=True); p.add_argument("--source",required=True); p.add_argument("--max_bytes",type=int,default=5_000_000_000); p.add_argument("--revision",default="main"); p.add_argument("--root",default="/data/mini-k3/data/raw"); a=p.parse_args()
 api=f"https://hf-mirror.com/api/datasets/{a.repo}/tree/{a.revision}?recursive=true&limit=1000"
 api_bytes = subprocess.check_output(["curl","-L","--fail","--retry","5","-A","Mozilla/5.0","--max-time","60",api])
 files=json.loads(api_bytes)
 blobs=[x for x in files if x.get("type")=="file" and Path(x.get("path","")).suffix.lower() in (".parquet",".jsonl",".json",".txt")]
 if not blobs: raise RuntimeError(f"no text files in {a.repo}")
 out=Path(a.root)/a.source; out.mkdir(parents=True,exist_ok=True); total=0; records=[]
 for item in blobs:
  size=int(item.get("size") or 0)
  if total and size and total+size>a.max_bytes: break
  rel=item["path"]; dst=out/rel; dst.parent.mkdir(parents=True,exist_ok=True)
  url=f"https://hf-mirror.com/datasets/{a.repo}/resolve/{a.revision}/{rel}"
  if dst.exists() and size and dst.stat().st_size==size: total+=size; records.append({"path":rel,"size":size,"url":url,"sha256":item.get("oid","")}); continue
  tmp=dst.with_suffix(dst.suffix+".incomplete")
  subprocess.run(["curl","-L","--fail","--retry","5","--retry-delay","2","-A","Mozilla/5.0","-o",str(tmp),url],check=True)
  if size and tmp.stat().st_size!=size: raise IOError(f"size mismatch {rel}")
  tmp.rename(dst); total+=dst.stat().st_size; records.append({"path":rel,"size":dst.stat().st_size,"url":url,"sha256":item.get("oid","")}); print(f"downloaded {rel} {dst.stat().st_size}")
 (out/"DOWNLOAD.json").write_text(json.dumps({"backend":"hf-mirror","repo":a.repo,"revision":a.revision,"bytes":total,"files":records},indent=2),encoding="utf-8")
if __name__=="__main__": main()
