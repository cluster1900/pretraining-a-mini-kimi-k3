"""Reconcile a stale controller status after an independently passed audit.

The historical controller can remain ``failed`` after its finalization process
exits, even when the repaired manifests, full audit, coverage report, and the
model smoke are all bound to the same bytes.  This command writes a new status
only with an explicit acknowledgement and records the superseded status hash.
It never downloads, tokenizes, or changes a shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile(data_root: Path, coverage_path: Path, acknowledge: bool) -> dict:
    if not acknowledge:
        raise ValueError("pass --acknowledge-audited-repair to write reconciled status")
    manifest_root = data_root / "manifests"
    status_path = data_root / "PIPELINE_STATUS.json"
    audit_path = manifest_root / "AUDIT.json"
    smoke_path = manifest_root / "SMOKE.json"
    train_path = manifest_root / "pretrain_stable.json"
    validation_path = manifest_root / "validation.json"
    required = [status_path, audit_path, smoke_path, train_path, validation_path, coverage_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing reconciliation evidence: " + ", ".join(missing))
    old = read(status_path)
    audit = read(audit_path)
    train = read(train_path)
    validation = read(validation_path)
    smoke = read(smoke_path)
    coverage = read(coverage_path)
    if audit.get("status") != "passed" or audit.get("train_validation_overlap") != 0:
        raise ValueError("AUDIT.json is not a passed zero-overlap audit")
    hashes = audit.get("manifest_sha256", {})
    if hashes.get(train_path.name) != digest(train_path) or hashes.get(validation_path.name) != digest(validation_path):
        raise ValueError("manifest hashes do not match AUDIT.json")
    if smoke.get("status") != "passed" or smoke.get("manifest_sha256") != digest(train_path):
        raise ValueError("SMOKE.json is not bound to the audited training manifest")
    if smoke.get("audit_sha256") != digest(audit_path):
        raise ValueError("SMOKE.json is not bound to the current AUDIT.json")
    if train.get("metadata", {}).get("status") != "audited" or validation.get("metadata", {}).get("status") != "audited":
        raise ValueError("manifests are not marked audited")
    if coverage.get("status") != "sufficient_fixed_mix" or coverage.get("deficits"):
        raise ValueError("coverage evidence is not sufficient_fixed_mix")
    if Path(str(coverage.get("root", data_root))).resolve() != data_root.resolve():
        raise ValueError("coverage root does not match data root")
    result = {
        "status": "complete",
        "stage": "reconciled_audited_finalization",
        "updated_at": time.time(),
        "reconciled_from": {
            "status": old.get("status"),
            "stage": old.get("stage"),
            "sha256": digest(status_path),
        },
        "evidence": {
            "audit_sha256": digest(audit_path),
            "train_manifest_sha256": digest(train_path),
            "validation_manifest_sha256": digest(validation_path),
            "smoke_sha256": digest(smoke_path),
            "coverage_sha256": digest(coverage_path),
            "report_archive": old.get("report_archive"),
        },
        "note": "Controller failure was superseded only after the repaired final manifests, audit, coverage, and model smoke were independently verified.",
    }
    fd, tmp = tempfile.mkstemp(prefix=".PIPELINE_STATUS.", dir=str(data_root), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, status_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--acknowledge-audited-repair", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconcile(args.data_root, args.coverage, args.acknowledge_audited_repair), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
