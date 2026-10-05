"""Regression tests for the pretraining readiness gate."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.readiness import ReadinessError, check_training_readiness, check_validation_readiness
from train.model_fingerprint import model_code_fingerprint


class ReadinessTests(unittest.TestCase):
    def _fixture(self, root: Path, pipeline_status: str = "complete",
                 data_root_name: str = "prepared-v2-supplement-v2") -> tuple[Path, Path, Path]:
        data_root = root / data_root_name
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
            "model_code_sha256": model_code_fingerprint(),
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

    def test_smoke_must_be_bound_to_current_model_code(self):
        with tempfile.TemporaryDirectory() as td:
            train, validation, coverage = self._fixture(Path(td))
            smoke_path = Path(train).parent / "SMOKE.json"
            smoke = json.loads(smoke_path.read_text())
            kwargs = dict(vocab_size=8, coverage_path=coverage)
            self.assertEqual(check_training_readiness(train, validation, **kwargs)["status"], "ready")

            smoke.pop("model_code_sha256")
            smoke_path.write_text(json.dumps(smoke))
            with self.assertRaisesRegex(ReadinessError, "model code changed since SMOKE.json"):
                check_training_readiness(train, validation, **kwargs)

            smoke["model_code_sha256"] = "0" * 64
            smoke_path.write_text(json.dumps(smoke))
            with self.assertRaisesRegex(ReadinessError, "re-run smoke_from_manifest.py"):
                check_training_readiness(train, validation, **kwargs)
            # An explicit expected fingerprint is honoured.
            result = check_training_readiness(train, validation, model_code_sha256="0" * 64, **kwargs)
            self.assertEqual(result["status"], "ready")

    def test_model_fingerprint_tracks_model_code(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ("config.py", "engine/muon.py", "engine/balancer.py", "models/a.py", "kernels/k.py"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {rel}\n")
            (root / "train.py").write_text("# not model code\n")
            first = model_code_fingerprint(root)
            (root / "train.py").write_text("# edited\n")
            self.assertEqual(model_code_fingerprint(root), first)
            (root / "models" / "a.py").write_text("# edited\n")
            self.assertNotEqual(model_code_fingerprint(root), first)
            (root / "models" / "a.py").write_text("# models/a.py\n")
            self.assertEqual(model_code_fingerprint(root), first)
            (root / "engine" / "balancer.py").write_text("# edited\n")
            self.assertNotEqual(model_code_fingerprint(root), first)
            (root / "engine" / "balancer.py").write_text("# engine/balancer.py\n")
            self.assertEqual(model_code_fingerprint(root), first)
            (root / "kernels" / "new.py").write_text("x = 1\n")
            self.assertNotEqual(model_code_fingerprint(root), first)

    def test_coverage_must_name_this_root(self):
        with tempfile.TemporaryDirectory() as td:
            train, validation, coverage = self._fixture(Path(td))
            data = json.loads(coverage.read_text())
            data.pop("root")
            coverage.write_text(json.dumps(data))
            with self.assertRaisesRegex(ReadinessError, "does not name its data root"):
                check_training_readiness(train, validation, vocab_size=8, coverage_path=coverage)

    def test_no_fallback_to_another_dataset_report(self):
        with tempfile.TemporaryDirectory() as td:
            # Data root prepared-v2-other has no report of its own; the
            # reports/supplement-v2 report next to it must not be picked up.
            train, validation, coverage = self._fixture(Path(td), data_root_name="prepared-v2-other")
            self.assertEqual(coverage.parent.name, "supplement-v2")
            with self.assertRaisesRegex(ReadinessError, "No coverage.json"):
                check_training_readiness(train, validation, vocab_size=8)
            # The matching report name is still discovered automatically.
            train, validation, coverage = self._fixture(Path(td) / "b")
            result = check_training_readiness(train, validation, vocab_size=8)
            self.assertEqual(Path(result["coverage"]["path"]), coverage.resolve())

    def test_validation_gate_binds_audit_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as td:
            train, validation, _ = self._fixture(Path(td))
            result = check_validation_readiness(validation, vocab_size=8, tokenizer_fingerprint="fixture")
            self.assertEqual(result["status"], "ready")
            with self.assertRaises(ReadinessError):
                check_validation_readiness(validation, vocab_size=8, tokenizer_fingerprint="other")


if __name__ == "__main__":
    unittest.main()
