"""
Dataset Preprocessing & Binary Shard Builder for Mini Kimi K3.
Features:
- Encodes text documents using K3Tokenizer with literal '[EOS]' protection
- Formats token streams into little-endian uint32 arrays (.bin)
- Writes fixed-token chunked shards (e.g., 50M tokens per shard)
- Generates the official manifest.json compatible with MultiSourceDataLoader
- Emits token statistics and data audit reports
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.data.tokenizer import K3Tokenizer


def build_shards_from_documents(
    source_name: str,
    doc_paths: List[Path],
    output_dir: Path,
    tokens_per_shard: int = 50_000_000,
    text_key: str = "text",
    tokenizer_model_path: str = None,
) -> Dict[str, Any]:
    """
    Reads documents from jsonl or text files, tokenizes with K3Tokenizer,
    and writes contiguous uint32 binary shards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = K3Tokenizer(model_path=tokenizer_model_path)

    shard_idx = 0
    buffer = []
    shard_paths = []
    total_tokens = 0
    total_docs = 0

    print(f"[*] Processing source: '{source_name}' across {len(doc_paths)} input file(s)...")

    for file_path in doc_paths:
        print(f"    -> Reading: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Extract text: JSONL or raw line
                if line.startswith("{") and line.endswith("}"):
                    try:
                        record = json.loads(line)
                        text = record.get(text_key, "")
                    except Exception:
                        text = line
                else:
                    text = line

                if not text:
                    continue

                # Tokenize with append_eos=True (token 163585)
                # allow_special_tokens=False is handled internally by K3Tokenizer
                tokens = tokenizer.encode(text, append_eos=True)
                buffer.extend(tokens)
                total_docs += 1
                total_tokens += len(tokens)

                # Flush shard when buffer threshold is reached
                while len(buffer) >= tokens_per_shard:
                    shard_data = np.array(buffer[:tokens_per_shard], dtype=np.uint32)
                    shard_file = output_dir / f"{source_name}_shard_{shard_idx:05d}.bin"
                    shard_data.tofile(shard_file)
                    shard_paths.append(str(shard_file.resolve()))
                    print(f"       Saved shard {shard_idx:05d}: {shard_file.name} ({tokens_per_shard:,} tokens)")
                    shard_idx += 1
                    buffer = buffer[tokens_per_shard:]

    # Flush remaining tokens in final shard if any
    if buffer:
        shard_data = np.array(buffer, dtype=np.uint32)
        shard_file = output_dir / f"{source_name}_shard_{shard_idx:05d}.bin"
        shard_data.tofile(shard_file)
        shard_paths.append(str(shard_file.resolve()))
        print(f"       Saved final shard {shard_idx:05d}: {shard_file.name} ({len(buffer):,} tokens)")
        shard_idx += 1

    report = {
        "source_name": source_name,
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "num_shards": len(shard_paths),
        "shards": shard_paths,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Build uint32 binary token shards for Mini K3")
    parser.add_argument("--source_name", type=str, required=True, help="Name of data source (e.g., fineweb-edu, code-python)")
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="Raw input .jsonl or .txt files")
    parser.add_argument("--output_dir", type=str, default="/data/mini-k3/data/tokenized", help="Directory to save uint32 shards")
    parser.add_argument("--manifest_path", type=str, default="/data/mini-k3/data/manifest.json", help="Path to output or update manifest.json")
    parser.add_argument("--tokens_per_shard", type=int, default=50_000_000, help="Tokens per binary shard (default 50M)")
    parser.add_argument("--weight", type=float, default=1.0, help="Sampling weight for this source in manifest")
    parser.add_argument("--text_key", type=str, default="text", help="Field name for text in JSONL")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Directory containing tiktoken.model")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_files]
    missing = [str(p) for p in input_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files missing: {missing}")

    out_dir = Path(args.output_dir) / args.source_name
    report = build_shards_from_documents(
        source_name=args.source_name,
        doc_paths=input_paths,
        output_dir=out_dir,
        tokens_per_shard=args.tokens_per_shard,
        text_key=args.text_key,
        tokenizer_model_path=args.tokenizer_model,
    )

    # Update or create manifest.json
    manifest_p = Path(args.manifest_path)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {"sources": {}}
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    manifest_data["sources"][args.source_name] = {
        "weight": args.weight,
        "shards": report["shards"],
        "total_tokens": report["total_tokens"],
        "total_docs": report["total_documents"],
    }

    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f" Sharding complete for source '{args.source_name}'!")
    print(f" - Documents Processed: {report['total_documents']:,}")
    print(f" - Tokens Processed:    {report['total_tokens']:,}")
    print(f" - Shards Produced:     {report['num_shards']}")
    print(f" - Manifest updated at: {manifest_p.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
