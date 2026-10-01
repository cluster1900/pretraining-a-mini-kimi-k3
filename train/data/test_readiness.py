"""Regression tests for the pretraining readiness gate."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.readiness import ReadinessError, check_training_readiness, check_validation_readiness


class ReadinessTests(unittest.TestCase):
    def _fixture(self, root: Path, pipeline_status: str = "complete") -> tuple[Path, Path, Path]:
        data_root = root / "prepared-v2-supplement-v2"
        manifest_root = data_root / "manifests"
        manifest_root.mkdir(parents=True)
        (data_root / "PIPELINE_STATUS.json").write_text(json.dumps({"status": pipeline_status}))
        report_root = root / "reports" / "supplement-v2"
        report_root.mkdir(parents=True)
        coverage = report_root / "coverage.json"

        mix = {
            "fineweb-edu": 0.45,
            "chinese-fineweb-edu": 0.15,
            "dolma-body": 0.10,
            "finemath": 0.05,
            "open-web-math": 0.05,
            "code-python": 0.10,
            "cosmopedia": 0.10,
        }
        coverage.write_text(json.dumps({
            "status": "sufficient_fixed_mix", "deficits": {},
            "root": str(data_root), "total_target_tokens": 7,
            "required_tokens": {name: 1 for name in mix},
            "available_train_tokens": {name: 1 for name in mix},
        }))
        sources = {}
        for i, (name, weight) in enumerate(mix.items()):
            shard = data_root / f"{name}.bin"
            shard.write_bytes((i + 1).to_bytes(4, "little"))
            sources[name] = {
                "weight": weight,
                "shards": [str(shard)],
                "shard_metadata": [{"path": str(shard), "tokens": 1, "bytes": 4}],
                "total_tokens": 1,
            }
        train = manifest_root / "pretrain_stable.json"
        validation = manifest_root / "validation.json"
        train.write_text(json.dumps({
            "metadata": {"schema_version": 2, "status": "audited", "split": "train", "vocab_size": 8,
                          "dtype": "<u4", "tokenizer_fingerprint": "fixture"},
            "sources": sources,
        }))
        val_shard = data_root / "validation.bin"
        val_shard.write_bytes((1).to_bytes(4, "little"))
        validation.write_text(json.dumps({
            "metadata": {"schema_version": 2, "status": "audited", "split": "validation", "vocab_size": 8,
                          "dtype": "<u4", "tokenizer_fingerprint": "fixture"},
            "sources": {"fineweb-edu": {
                "weight": 0.45,
                "shards": [str(val_shard)],
                "shard_metadata": [{"path": str(val_shard), "tokens": 1, "bytes": 4}],
                "total_tokens": 1,
            }},
        }))
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train, validation)}
        audit_path = manifest_root / "AUDIT.json"
        audit_path.write_text(json.dumps({
            "status": "passed", "train_validation_overlap": 0, "manifest_sha256": hashes,
            "tokenizer_fingerprint": "fixture",
        }))
        (manifest_root / "SMOKE.json").write_text(json.dumps({
            "status": "passed", "manifest_sha256": hashes[train.name],
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        }))
        return train, validation, coverage

    def test_passed_audit_and_status_are_required(self):
        with tempfile.TemporaryDirectory() as td:
            train, validation, coverage = self._fixture(Path(td))
            result = check_training_readiness(
                train, validation, vocab_size=8,
                stable_mix={
                    "fineweb-edu": 0.45, "chinese-fineweb-edu": 0.15,
                    "dolma-body": 0.10, "finemath": 0.05,
                    "open-web-math": 0.05, "code-python": 0.10,
                    "cosmopedia": 0.10,
                }, coverage_path=coverage,
                expected_training_tokens=7,
            )
            self.assertEqual(result["status"], "ready")

            # A short smoke run consumes a prefix of the audited 10B recipe;
            # the gate must require enough coverage without requiring the
            # short run's token budget to equal the full recipe target.
            result = check_training_readiness(
                train, validation, vocab_size=8, coverage_path=coverage,
                expected_training_tokens=3,
            )
            self.assertEqual(result["status"], "ready")

            with self.assertRaises(ReadinessError):
                check_training_readiness(
                    train, validation, vocab_size=8, coverage_path=coverage,
                    expected_training_tokens=8,
                )

            pipeline = Path(train).parent.parent / "PIPELINE_STATUS.json"
            pipeline.write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(ReadinessError):
                check_training_readiness(train, validation, vocab_size=8, coverage_path=coverage)

            coverage.write_text(json.dumps({
                "status": "sufficient_fixed_mix", "deficits": {},
                "root": str(Path(td) / "other"), "total_target_tokens": 7,
                "required_tokens": {name: 1 for name in (
                    "fineweb-edu", "chinese-fineweb-edu", "dolma-body", "finemath",
                    "open-web-math", "code-python", "cosmopedia")},
                "available_train_tokens": {name: 1 for name in (
                    "fineweb-edu", "chinese-fineweb-edu", "dolma-body", "finemath",
                    "open-web-math", "code-python", "cosmopedia")},
            }))
            with self.assertRaises(ReadinessError):
                check_training_readiness(train, validation, vocab_size=8, coverage_path=coverage,
                                         require_pipeline_complete=False)

    def test_validation_gate_binds_audit_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as td:
            train, validation, _ = self._fixture(Path(td))
            result = check_validation_readiness(validation, vocab_size=8, tokenizer_fingerprint="fixture")
            self.assertEqual(result["status"], "ready")
            with self.assertRaises(ReadinessError):
                check_validation_readiness(validation, vocab_size=8, tokenizer_fingerprint="other")


if __name__ == "__main__":
    unittest.main()
