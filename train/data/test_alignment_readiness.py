import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.alignment_readiness import AlignmentReadinessError, check_alignment_jsonl


class AlignmentReadinessTests(unittest.TestCase):
    def test_file_must_match_audited_hash_and_kind(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "train.jsonl"
            data.write_text('{"input_ids":[1,2]}\n', encoding="utf-8")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            manifest = root / "alignment.json"
            manifest.write_text(json.dumps({
                "metadata": {"schema_version": 2, "status": "audited",
                              "vocab_size": 8, "tokenizer_fingerprint": "fp"},
                "sources": {"demo": {"status": "complete", "kind": "sft",
                    "tokenizer_fingerprint": "fp", "files": [{
                        "path": str(data), "split": "train", "bytes": data.stat().st_size,
                        "sha256": digest,
                    }]}}
            }))
            evidence = check_alignment_jsonl(manifest, data, "sft", 8)
            self.assertEqual(evidence["status"], "ready")
            with self.assertRaises(AlignmentReadinessError):
                check_alignment_jsonl(manifest, data, "rm", 8)
            data.write_text('{"input_ids":[3]}\n', encoding="utf-8")
            with self.assertRaises(AlignmentReadinessError):
                check_alignment_jsonl(manifest, data, "sft", 8)


if __name__ == "__main__":
    unittest.main()
