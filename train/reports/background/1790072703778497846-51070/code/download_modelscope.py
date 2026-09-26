"""ModelScope-native raw file downloader (fallback when HF API is blocked)."""
import argparse, json
from pathlib import Path
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import dataset_file_download

DEFAULT = {
 "fineweb-edu":"HuggingFaceFW/fineweb-edu",
 "culturax":"opencsg/chinese-fineweb-edu",
 "cosmopedia":"AI-ModelScope/chinese-cosmopedia",
 "code-python":"HuggingFaceCode/stack-v3-train",
 "mathr":"modelscope/MathR",
 "numinamath":"AI-MO/NuminaMath-CoT"
 ,"skypile":"AI-ModelScope/SkyPile-150B"
}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--repo",default=None); p.add_argument("--max_bytes",type=int,default=5_000_000_000); p.add_argument("--output_root",default="/data/mini-k3/data/raw"); args=p.parse_args()
 repo=args.repo or DEFAULT.get(args.source)
 if not repo: raise ValueError("--repo is required for unknown source")
 files=HubApi().get_dataset_files(repo,recursive=True)
 blobs=[x for x in files if x.get("Type")=="blob" and (Path(x["Path"]).suffix.lower() in (".parquet",".jsonl",".json",".txt") or x["Path"].lower().endswith((".json.gz",".jsonl.gz")))]
 if not blobs: raise RuntimeError(f"No downloadable text files found in {repo}")
 out=Path(args.output_root)/args.source; out.mkdir(parents=True,exist_ok=True); total=0; records=[]
 for item in blobs:
  size=int(item.get("Size") or 0)
  if total and size and total+size>args.max_bytes: break
  local=dataset_file_download(repo,item["Path"],local_dir=str(out))
  records.append({"repo":repo,"path":item["Path"],"local":local,"size":size,"sha256":item.get("Sha256","")}); total+=size
  print(f"downloaded {item['Path']} ({size} bytes)")
 (out/"DOWNLOAD.json").write_text(json.dumps({"backend":"modelscope","source":args.source,"files":records,"bytes":total},indent=2),encoding="utf-8")
if __name__=="__main__": main()
