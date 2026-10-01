"""Fail-closed checks required before a pretraining process may start.

The data pipeline writes a passed ``AUDIT.json`` and a manifest-bound model
smoke report. The training entry point verifies those bindings instead of
accepting any JSON file that happens to contain readable shards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ReadinessError(RuntimeError):
    """Raised when a training input is not bound to a passed audit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReadinessError(f"Missing readiness file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"Cannot read JSON readiness file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"Readiness file is not a JSON object: {path}")
    return value


def _coverage_path(root: Path) -> Path | None:
    candidates = [root / "coverage.json"]
    name = root.name
    if name.startswith("prepared-v2-"):
        candidates.append(root.parent / "reports" / name.removeprefix("prepared-v2-") / "coverage.json")
    # Never fall back to another dataset version.  A stale report can make a
    # complete-looking manifest appear to have enough tokens for this run.
    candidates.append(root.parent / "reports" / "supplement-v2" / "coverage.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _check_manifest(
    manifest_path: Path,
    expected_split: str,
    expected_mix: Mapping[str, float] | None,
    vocab_size: int,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ReadinessError(f"{manifest_path}: missing metadata")
    if metadata.get("schema_version") != 2 or metadata.get("status") != "audited":
        raise ReadinessError(f"{manifest_path}: manifest is not schema-v2 audited data")
    if metadata.get("split") != expected_split:
        raise ReadinessError(
            f"{manifest_path}: expected split {expected_split!r}, got {metadata.get('split')!r}"
        )
    if metadata.get("vocab_size") != vocab_size:
        raise ReadinessError(f"{manifest_path}: vocab size does not match model config")
    if metadata.get("dtype") != "<u4" or not metadata.get("tokenizer_fingerprint"):
        raise ReadinessError(f"{manifest_path}: tokenizer fingerprint/dtype binding is missing")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ReadinessError(f"{manifest_path}: no sources")
    if expected_mix is not None and expected_split == "train":
        if set(sources) != set(expected_mix):
            raise ReadinessError(f"{manifest_path}: source set differs from configured stable mix")
        for name, target in expected_mix.items():
            actual = float(sources[name].get("weight", -1.0))
            if abs(actual - float(target)) > 1e-6:
                raise ReadinessError(f"{manifest_path}: weight for {name} is {actual}, expected {target}")

    total_tokens = 0
    for source, info in sources.items():
        if not isinstance(info, dict):
            raise ReadinessError(f"{manifest_path}: malformed source {source}")
        shards = info.get("shards")
        metadata_rows = info.get("shard_metadata")
        if not isinstance(shards, list) or not shards:
            raise ReadinessError(f"{manifest_path}: {source} has no shards")
        if not isinstance(metadata_rows, list) or len(metadata_rows) != len(shards):
            raise ReadinessError(f"{manifest_path}: {source} shard metadata is incomplete")
        rows_by_path = {str(row.get("path")): row for row in metadata_rows if isinstance(row, dict)}
        source_tokens = 0
        for raw_path in shards:
            path = Path(raw_path)
            row = rows_by_path.get(str(path))
            if row is None:
                raise ReadinessError(f"{manifest_path}: missing shard metadata for {path}")
            if not path.is_file():
                raise ReadinessError(f"{manifest_path}: missing shard {path}")
            tokens = int(row.get("tokens", 0))
            expected_bytes = int(row.get("bytes", -1))
            if tokens <= 0 or expected_bytes != tokens * 4 or path.stat().st_size != expected_bytes:
                raise ReadinessError(f"{manifest_path}: shard size metadata mismatch for {path}")
            source_tokens += tokens
            total_tokens += tokens
        if info.get("total_tokens") is not None and int(info["total_tokens"]) != source_tokens:
            raise ReadinessError(f"{manifest_path}: source total_tokens mismatch for {source}")

        expected_total = int(manifest.get("total_tokens", total_tokens))
    if expected_total != total_tokens:
        raise ReadinessError(f"{manifest_path}: total_tokens does not equal shard metadata")
    return {"path": str(manifest_path), "sha256": _sha256(manifest_path), "tokens": total_tokens}


def check_training_readiness(
    manifest_path: str | Path,
    validation_manifest_path: str | Path,
    *,
    vocab_size: int,
    stable_mix: Mapping[str, float] | None = None,
    require_pipeline_complete: bool = True,
    coverage_path: str | Path | None = None,
    expected_parameters: Mapping[str, int] | None = None,
    expected_training_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate manifests, audit bindings, smoke bindings, and coverage."""

    manifest = Path(manifest_path).resolve()
    validation = Path(validation_manifest_path).resolve()
    if manifest.parent != validation.parent:
        raise ReadinessError("Training and validation manifests must come from the same audited directory")
    audit_path = manifest.parent / "AUDIT.json"
    audit = _read_json(audit_path)
    if audit.get("status") != "passed" or audit.get("train_validation_overlap") != 0:
        raise ReadinessError(f"{audit_path}: full manifest audit is not passed")
    manifest_hashes = audit.get("manifest_sha256")
    if not isinstance(manifest_hashes, dict):
        raise ReadinessError(f"{audit_path}: manifest hash bindings are missing")

    train_info = _check_manifest(manifest, "train", stable_mix, vocab_size)
    validation_info = _check_manifest(validation, "validation", None, vocab_size)
    train_meta = _read_json(manifest).get("metadata", {})
    validation_meta = _read_json(validation).get("metadata", {})
    audit_fingerprint = audit.get("tokenizer_fingerprint")
    if audit_fingerprint and (
        train_meta.get("tokenizer_fingerprint") != audit_fingerprint
        or validation_meta.get("tokenizer_fingerprint") != audit_fingerprint
    ):
        raise ReadinessError("Manifest tokenizer fingerprint does not match AUDIT.json")
    for path, info in ((manifest, train_info), (validation, validation_info)):
        if manifest_hashes.get(path.name) != info["sha256"]:
            raise ReadinessError(f"{path}: hash does not match AUDIT.json")

    smoke_path = manifest.parent / "SMOKE.json"
    smoke = _read_json(smoke_path)
    if smoke.get("status") != "passed" or smoke.get("manifest_sha256") != train_info["sha256"]:
        raise ReadinessError(f"{smoke_path}: model smoke is missing or bound to another manifest")
    if smoke.get("audit_sha256") != _sha256(audit_path):
        raise ReadinessError(f"{smoke_path}: model smoke is bound to a different AUDIT.json")
    if expected_parameters is not None:
        actual_parameters = smoke.get("parameters")
        if not isinstance(actual_parameters, dict):
            raise ReadinessError(f"{smoke_path}: model parameter signature is missing")
        for field, expected in expected_parameters.items():
            if int(actual_parameters.get(field, -1)) != int(expected):
                raise ReadinessError(
                    f"{smoke_path}: {field}={actual_parameters.get(field)!r}, expected {expected}"
                )

    pipeline_path = manifest.parent.parent / "PIPELINE_STATUS.json"
    pipeline = _read_json(pipeline_path)
    if require_pipeline_complete and pipeline.get("status") != "complete":
        raise ReadinessError(
            f"{pipeline_path}: pipeline status is {pipeline.get('status')!r}; "
            "repair/re-run the audited finalization before training"
        )

    selected_coverage = Path(coverage_path).resolve() if coverage_path else _coverage_path(manifest.parent.parent)
    if selected_coverage is None:
        raise ReadinessError("No coverage.json found for the audited data root")
    coverage = _read_json(selected_coverage)
    if coverage.get("root") and Path(str(coverage["root"])).resolve() != manifest.parent.parent:
        raise ReadinessError(f"{selected_coverage}: coverage root is not this manifest's data root")
    if coverage.get("status") != "sufficient_fixed_mix" or coverage.get("deficits"):
        raise ReadinessError(f"{selected_coverage}: fixed-mix coverage is not sufficient")
    if stable_mix is not None:
        required = coverage.get("required_tokens")
        available = coverage.get("available_train_tokens")
        if not isinstance(required, dict) or not isinstance(available, dict):
            raise ReadinessError(f"{selected_coverage}: per-source coverage evidence is missing")
        if set(required) != set(stable_mix) or any(int(available.get(name, 0)) < int(required[name]) for name in stable_mix):
            raise ReadinessError(f"{selected_coverage}: per-source fixed-mix coverage is insufficient")
    if expected_training_tokens is not None and int(coverage.get("total_target_tokens", -1)) < int(expected_training_tokens):
        raise ReadinessError(
            f"{selected_coverage}: total_target_tokens={coverage.get('total_target_tokens')!r} "
            f"is below the requested run budget {expected_training_tokens}"
        )

    return {
        "status": "ready",
        "manifest": train_info,
        "validation": validation_info,
        "audit": str(audit_path),
        "smoke": str(smoke_path),
        "pipeline": str(pipeline_path),
        "coverage": {"path": str(selected_coverage), "status": coverage.get("status")},
    }


def check_validation_readiness(
    validation_manifest_path: str | Path,
    *,
    vocab_size: int,
    tokenizer_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate a held-out manifest before read-only evaluation.

    Evaluation does not need the training coverage or model smoke gate, but it
    must still consume the same audited validation split, manifest hash, shard
    metadata, and tokenizer binding. Keeping this smaller gate separate avoids
    making evaluation depend on a particular training run while preventing an
    arbitrary JSON file from being called a validation set.
    """
    validation = Path(validation_manifest_path).resolve()
    root = validation.parent
    audit_path = root / "AUDIT.json"
    audit = _read_json(audit_path)
    if audit.get("status") != "passed" or audit.get("train_validation_overlap") != 0:
        raise ReadinessError(f"{audit_path}: full manifest audit is not passed")
    manifest_hashes = audit.get("manifest_sha256")
    if not isinstance(manifest_hashes, dict):
        raise ReadinessError(f"{audit_path}: manifest hash bindings are missing")
    info = _check_manifest(validation, "validation", None, vocab_size)
    if manifest_hashes.get(validation.name) != info["sha256"]:
        raise ReadinessError(f"{validation}: hash does not match AUDIT.json")
    metadata = _read_json(validation).get("metadata", {})
    audit_fingerprint = audit.get("tokenizer_fingerprint")
    if audit_fingerprint and metadata.get("tokenizer_fingerprint") != audit_fingerprint:
        raise ReadinessError(f"{validation}: tokenizer fingerprint does not match AUDIT.json")
    if tokenizer_fingerprint and metadata.get("tokenizer_fingerprint") != tokenizer_fingerprint:
        raise ReadinessError(f"{validation}: tokenizer fingerprint does not match tokenizer assets")
    return {
        "status": "ready",
        "validation": info,
        "audit": str(audit_path),
        "tokenizer_fingerprint": metadata.get("tokenizer_fingerprint"),
    }
