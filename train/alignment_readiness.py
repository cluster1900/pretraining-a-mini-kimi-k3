"""Fail-closed provenance checks for SFT and preference JSONL inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class AlignmentReadinessError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_alignment_jsonl(manifest_path: str | Path, jsonl_path: str | Path,
                          mode: str, vocab_size: int) -> dict:
    manifest_path = Path(manifest_path).resolve()
    jsonl_path = Path(jsonl_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlignmentReadinessError(f"cannot read alignment manifest: {manifest_path}: {exc}") from exc
    meta = manifest.get("metadata", {})
    if meta.get("schema_version") != 2 or meta.get("status") != "audited":
        raise AlignmentReadinessError("alignment manifest is not schema-v2 audited")
    if meta.get("vocab_size") != vocab_size or not meta.get("tokenizer_fingerprint"):
        raise AlignmentReadinessError("alignment tokenizer/vocab binding is missing")
    wanted_kind = "sft" if mode == "sft" else "preference"
    sources = manifest.get("sources", {})
    matches = []
    for source, info in sources.items():
        if info.get("status") != "complete" or info.get("kind") != wanted_kind:
            continue
        if info.get("tokenizer_fingerprint") != meta["tokenizer_fingerprint"]:
            raise AlignmentReadinessError(f"{source}: tokenizer fingerprint mismatch")
        for item in info.get("files", []):
            if item.get("split") == "train" and Path(item.get("path", "")).resolve() == jsonl_path:
                matches.append((source, item))
    if len(matches) != 1:
        raise AlignmentReadinessError(
            f"{jsonl_path} is not the unique audited {wanted_kind} train file in {manifest_path}"
        )
    source, item = matches[0]
    if not jsonl_path.is_file() or jsonl_path.stat().st_size != int(item.get("bytes", -1)):
        raise AlignmentReadinessError(f"alignment file size changed: {jsonl_path}")
    actual_hash = _digest(jsonl_path)
    if actual_hash != item.get("sha256"):
        raise AlignmentReadinessError(f"alignment file hash changed: {jsonl_path}")
    return {
        "status": "ready", "source": source, "kind": wanted_kind,
        "manifest": str(manifest_path), "manifest_sha256": _digest(manifest_path),
        "jsonl": str(jsonl_path), "jsonl_sha256": actual_hash,
        "tokenizer_fingerprint": meta["tokenizer_fingerprint"],
    }
