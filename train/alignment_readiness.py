"""Fail-closed provenance checks for SFT and preference JSONL inputs.

Mode -> accepted manifest kind:

* ``sft``          -> ``sft``
* ``rm`` / ``dpo`` -> ``preference`` (needs ``chosen_ids``/``rejected_ids``)
* ``ppo``          -> ``preference`` (``prompt_ids``) or ``sft`` (prompt part only)
* ``grpo``         -> ``sft`` only. GRPO uses the prompt part of each SFT row
  (tokens before the first supervised label) and a gold answer string.

Gold answers: tokenized SFT rows (``tokenize_v2.encode_source``) store only
``id, source, split, split_group, content_sha256, input_ids, labels``; no answer
string. The audited OpenR1 rows one stage earlier
(``<root>/decontaminated/openr1/part-*.jsonl``, written by the canonical ->
clean -> dedup -> near-dedup -> contamination chain) still carry the upstream
``answer`` field. ``load_audited_gold_answers`` joins on ``id`` and verifies the
chain: the decontaminated ``COMPLETE.json`` digest must equal the
``upstream_sha256`` that the audited alignment manifest records for that
source, and every part must match its recorded size and SHA256. Nothing is
loaded from unaudited files. Sources without an ``answer`` field (OpenAssistant,
OpenHermes) fail loudly. Note: contamination_v2 scans message text, not the
``answer`` field itself; for OpenR1 the answer also appears inside the scanned
assistant message.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class AlignmentReadinessError(RuntimeError):
    pass


MODE_KINDS = {
    "sft": ("sft",),
    "rm": ("preference",),
    "dpo": ("preference",),
    "ppo": ("preference", "sft"),
    "grpo": ("sft",),
}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_manifest(manifest_path: Path, vocab_size: int) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlignmentReadinessError(f"cannot read alignment manifest: {manifest_path}: {exc}") from exc
    meta = manifest.get("metadata", {})
    if meta.get("schema_version") != 2 or meta.get("status") != "audited":
        raise AlignmentReadinessError("alignment manifest is not schema-v2 audited")
    if meta.get("vocab_size") != vocab_size or not meta.get("tokenizer_fingerprint"):
        raise AlignmentReadinessError("alignment tokenizer/vocab binding is missing")
    return manifest


def check_alignment_jsonl(manifest_path: str | Path, jsonl_path: str | Path,
                          mode: str, vocab_size: int) -> dict:
    manifest_path = Path(manifest_path).resolve()
    jsonl_path = Path(jsonl_path).resolve()
    if mode not in MODE_KINDS:
        raise AlignmentReadinessError(f"unknown alignment mode {mode!r}")
    manifest = _read_manifest(manifest_path, vocab_size)
    meta = manifest["metadata"]
    wanted = MODE_KINDS[mode]
    sources = manifest.get("sources", {})
    matches = []
    for source, info in sources.items():
        if info.get("status") != "complete" or info.get("kind") not in wanted:
            continue
        if info.get("tokenizer_fingerprint") != meta["tokenizer_fingerprint"]:
            raise AlignmentReadinessError(f"{source}: tokenizer fingerprint mismatch")
        for item in info.get("files", []):
            if item.get("split") == "train" and Path(item.get("path", "")).resolve() == jsonl_path:
                matches.append((source, info, item))
    if len(matches) != 1:
        raise AlignmentReadinessError(
            f"{jsonl_path} is not the unique audited {'/'.join(wanted)} train file in {manifest_path}"
        )
    source, info, item = matches[0]
    if not jsonl_path.is_file() or jsonl_path.stat().st_size != int(item.get("bytes", -1)):
        raise AlignmentReadinessError(f"alignment file size changed: {jsonl_path}")
    actual_hash = _digest(jsonl_path)
    if actual_hash != item.get("sha256"):
        raise AlignmentReadinessError(f"alignment file hash changed: {jsonl_path}")
    return {
        "status": "ready", "source": source, "kind": info["kind"],
        "manifest": str(manifest_path), "manifest_sha256": _digest(manifest_path),
        "jsonl": str(jsonl_path), "jsonl_sha256": actual_hash,
        "tokenizer_fingerprint": meta["tokenizer_fingerprint"],
        "upstream_sha256": info.get("upstream_sha256"),
    }


def load_audited_gold_answers(evidence: dict, field: str = "answer",
                              decontaminated_marker: str | Path | None = None):
    """Return ``({row_id: answer}, report)`` for the train rows of an audited SFT source.

    ``evidence`` is the dict returned by ``check_alignment_jsonl``. The marker
    defaults to ``<root>/decontaminated/<source>/COMPLETE.json`` where the
    tokenized file is ``<root>/tokenized/<source>/train.jsonl``.
    """
    if evidence.get("kind") != "sft":
        raise AlignmentReadinessError("gold answers are only joined onto audited SFT rows")
    source = evidence["source"]
    upstream = evidence.get("upstream_sha256")
    if not upstream:
        raise AlignmentReadinessError(
            f"{source}: alignment manifest lacks upstream_sha256; cannot bind gold answers"
        )
    jsonl = Path(evidence["jsonl"])
    if decontaminated_marker is None:
        root = jsonl.parent.parent.parent
        marker = root / "decontaminated" / source / "COMPLETE.json"
    else:
        marker = Path(decontaminated_marker)
    if not marker.is_file():
        raise AlignmentReadinessError(f"{source}: audited upstream marker missing: {marker}")
    marker_hash = _digest(marker)
    if marker_hash != upstream:
        raise AlignmentReadinessError(
            f"{source}: {marker} does not match the upstream_sha256 bound in the alignment manifest"
        )
    report = json.loads(marker.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise AlignmentReadinessError(f"{source}: upstream stage is not complete")
    answers = {}
    rows = missing = 0
    for part in report.get("parts", []):
        path = Path(part["path"])
        if not path.is_file() or path.stat().st_size != int(part.get("bytes", -1)) \
                or _digest(path) != part.get("sha256"):
            raise AlignmentReadinessError(f"{source}: upstream part changed: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != "train":
                    continue
                rows += 1
                value = row.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = str(value)
                if isinstance(value, str) and value.strip():
                    answers[row["id"]] = value
                else:
                    missing += 1
    if not answers:
        raise AlignmentReadinessError(
            f"{source}: no audited train row carries a non-empty {field!r}; "
            "GRPO answer rewards need a source such as openr1"
        )
    return answers, {
        "marker": str(marker), "marker_sha256": marker_hash, "field": field,
        "train_rows": rows, "with_answer": len(answers), "without_answer": missing,
    }
