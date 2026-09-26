"""Streaming downloader for the fixed public Mini-K3 data quota.
Writes raw JSONL under /data/mini-k3/data/raw and never mirrors whole corpora.
"""
import argparse, json, time, os
from pathlib import Path

SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    "culturax": ("uonlp/CulturaX", "en", "train"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-3plus", "train"),
    "open-web-math": ("open-web-math/open-web-math", None, "train"),
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "web_samples_v2", "train"),
    "code-python": ("bigcode/starcoderdata", "python", "train"),
    "dolma": ("allenai/dolma", None, "train"),
}

BACKENDS = ["https://huggingface.co", "https://hf-mirror.com"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", choices=sorted(SOURCES), required=True)
    p.add_argument("--target_tokens", type=int, required=True); p.add_argument("--max_docs", type=int, default=0)
    p.add_argument("--output_root", default="/data/mini-k3/data/raw")
    args = p.parse_args(); repo, config, split = SOURCES[args.source]
    out = Path(args.output_root) / args.source; out.mkdir(parents=True, exist_ok=True)
    path = out / "part-00000.jsonl"; meta = out / "DOWNLOAD.json"
    done_tokens = done_docs = 0
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try: done_tokens += len(json.loads(line).get("text", "").split()); done_docs += 1
            except Exception: pass
    kwargs = {"path": repo, "split": split, "streaming": True}
    if config: kwargs["name"] = config
    errors = []
    ds = None
    for endpoint in BACKENDS:
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            from datasets import load_dataset
            ds = load_dataset(**kwargs)
            backend = endpoint
            break
        except Exception as exc:
            errors.append({"backend": endpoint, "error": repr(exc)})
    if ds is None:
        # ModelScope is an explicit fallback rather than an implicit alias:
        # repository IDs differ and must be supplied by a mapping file.
        mapping = Path(os.environ.get("MINIK3_MODELSCOPE_MAP", ""))
        ms_id = None
        if mapping.is_file():
            ms_id = json.loads(mapping.read_text()).get(args.source)
        if ms_id:
            try:
                from modelscope.msdatasets import MsDataset
                ds = MsDataset.load(ms_id, split=split, streaming=True)
                backend = "modelscope"
            except Exception as exc:
                errors.append({"backend": "modelscope", "error": repr(exc)})
        if ds is None:
            raise RuntimeError(json.dumps({"source": args.source, "errors": errors}, ensure_ascii=False))
    with path.open("a", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or row.get("content") or row.get("body") or ""
            if not isinstance(text, str) or len(text.strip()) < 40: continue
            words = len(text.split())
            if done_tokens + words > args.target_tokens: break
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n"); f.flush()
            done_tokens += words; done_docs += 1
            if args.max_docs and done_docs >= args.max_docs: break
    meta.write_text(json.dumps({"dataset": repo, "config": config, "split": split, "backend": backend,
        "documents": done_docs, "approx_words": done_tokens, "target_tokens": args.target_tokens,
        "completed_at": time.time()}, indent=2), encoding="utf-8")
    print(json.dumps({"source": args.source, "documents": done_docs, "approx_words": done_tokens}))

if __name__ == "__main__": main()
