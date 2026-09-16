"""Bind a functional smoke run to the exact manifest accepted by the full audit."""
import json
from pathlib import Path

from .canonical_v2 import digest_file


def verified_smoke_manifest(manifest, expected_weights, vocab_size):
    manifest = Path(manifest).resolve()
    audit_path = manifest.parent / 'AUDIT.json'
    audit = json.loads(audit_path.read_text())
    manifest_hash = digest_file(manifest)
    if audit.get('status') != 'passed':
        raise ValueError('Full data audit has not passed')
    if audit.get('manifest_sha256', {}).get(manifest.name) != manifest_hash:
        raise ValueError('Manifest is not bound to the passed full audit')
    data = json.loads(manifest.read_text())
    meta = data['metadata']
    if (meta.get('status') != 'audited' or meta.get('dtype') != '<u4'
            or meta.get('split') != 'train' or meta.get('vocab_size') != vocab_size
            or not meta.get('tokenizer_fingerprint')
            or meta['tokenizer_fingerprint'] != audit.get('tokenizer_fingerprint')):
        raise ValueError('Smoke manifest metadata mismatch')
    if set(data['sources']) != set(expected_weights):
        raise ValueError('Smoke manifest source/config mismatch')
    sampled = {}
    for source, weight in expected_weights.items():
        info = data['sources'][source]
        if info['weight'] != weight or not info['shards']:
            raise ValueError('Smoke manifest mixture/config mismatch: ' + source)
        shard = info['shards'][0]
        matches = [f for f in info['shard_metadata'] if f['path'] == shard]
        if len(matches) != 1:
            raise ValueError('Missing/ambiguous smoke shard metadata: ' + source)
        item = matches[0]
        path = Path(shard)
        if (item['bytes'] != item['tokens'] * 4 or path.stat().st_size != item['bytes']
                or digest_file(path) != item['sha256']):
            raise ValueError('Smoke shard changed since audit: ' + source)
        sampled[source] = {'path': shard, 'sha256': item['sha256']}
    return data, {'manifest_sha256': manifest_hash, 'audit_sha256': digest_file(audit_path),
                  'sampled_shards': sampled}
