"""Reject stale audits, edited mixtures, and mutated shards before GPU smoke."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.data.canonical_v2 import atomic_json, digest_file
from train.data.smoke_audit import verified_smoke_manifest


class SmokeAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.shard = root / 'shard.bin'
        self.shard.write_bytes(struct.pack('<II', 1, 2))
        self.manifest = root / 'pretrain_stable.json'
        self.audit = root / 'AUDIT.json'
        self.data = {
            'metadata': {'status': 'audited', 'dtype': '<u4', 'split': 'train',
                         'vocab_size': 300, 'tokenizer_fingerprint': 'fixture'},
            'sources': {'source': {'weight': 1.0, 'shards': [str(self.shard)],
                'shard_metadata': [{'path': str(self.shard), 'bytes': 8, 'tokens': 2,
                                    'sha256': digest_file(self.shard)}]}}}
        atomic_json(self.manifest, self.data)
        self.bind()

    def bind(self):
        atomic_json(self.audit, {'status': 'passed', 'tokenizer_fingerprint': 'fixture',
                    'manifest_sha256': {self.manifest.name: digest_file(self.manifest)}})

    def verify(self):
        return verified_smoke_manifest(self.manifest, {'source': 1.0}, 300)

    def test_passed_audit_binds_manifest_and_shard(self):
        data, evidence = self.verify()
        self.assertEqual(data, self.data)
        self.assertEqual(evidence['sampled_shards']['source']['sha256'], digest_file(self.shard))

    def test_unbound_legacy_audit_rejected(self):
        atomic_json(self.audit, {'status': 'passed'})
        with self.assertRaisesRegex(ValueError, 'not bound'): self.verify()

    def test_manifest_modified_after_audit_rejected(self):
        self.data['sources']['source']['weight'] = 0.5
        atomic_json(self.manifest, self.data)
        with self.assertRaisesRegex(ValueError, 'not bound'): self.verify()

    def test_audited_mixture_different_from_config_rejected(self):
        self.data['sources']['source']['weight'] = 0.5
        atomic_json(self.manifest, self.data)
        self.bind()
        with self.assertRaisesRegex(ValueError, 'mixture/config'): self.verify()

    def test_same_size_corrupted_shard_rejected(self):
        self.shard.write_bytes(struct.pack('<II', 1, 3))
        with self.assertRaisesRegex(ValueError, 'changed since audit'): self.verify()


if __name__ == '__main__': unittest.main()
