"""Measure available pretraining tokens against the documented WSD mixture."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.config import MiniK3Config
from train.data.canonical_v2 import atomic_json


def assess(root, report, total_tokens=10_000_007_168, buffer=0.20):
    root = Path(root)
    cfg = MiniK3Config()
    stable_fraction = 1.0 - cfg.decay_frac
    decay_fraction = cfg.decay_frac
    required = {source: total_tokens * (stable_fraction * cfg.stable_mix.get(source, 0.0)
                                        + decay_fraction * cfg.decay_mix.get(source, 0.0))
                for source in cfg.stable_mix}
    available = {}
    evidence = {}
    for source in required:
        marker = root / 'tokenized' / source / 'COMPLETE.json'
        if marker.is_file():
            data = json.loads(marker.read_text())
            if data.get('kind') == 'pretrain':
                available[source] = int(data['counts'].get('train_tokens', 0))
                evidence[source] = str(marker)
    deficits = {source: max(required[source] - available.get(source, 0), 0)
                for source in required if available.get(source, 0) < required[source]}
    shares = {s: stable_fraction * cfg.stable_mix.get(s, 0.0)
              + decay_fraction * cfg.decay_mix.get(s, 0.0) for s in required}
    max_fixed_tokens = min((available.get(source, 0) / share for source, share in shares.items()
                            if share > 0), default=0)
    data = {
        'status': 'sufficient_fixed_mix' if not deficits else 'insufficient_fixed_mix',
        'scope': 'Training token availability versus the documented stable/decay mixture; no download or training performed',
        'root': str(root), 'total_target_tokens': total_tokens,
        'stable_fraction': stable_fraction, 'decay_fraction': decay_fraction,
        'required_tokens': {s: int(v) for s, v in required.items()},
        'available_train_tokens': available, 'deficits': {s: int(v) for s, v in deficits.items()},
        'recommended_supplement_tokens_with_buffer': {s: int(v * (1 + buffer)) for s, v in deficits.items()},
        'max_tokens_without_reuse_at_fixed_mix': int(max_fixed_tokens),
        'evidence': evidence,
        'completion_scope': 'final_manifest' if (root / 'manifests/AUDIT.json').is_file()
                            else 'provisional_tokenized_reports',
        'warnings': [
            'A total-token surplus cannot substitute for a missing source while the mixture is fixed.',
            'This report does not authorize downloads or a change to the training mixture.',
        ],
    }
    atomic_json(report, data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--total-tokens', type=int, default=10_000_007_168)
    parser.add_argument('--buffer', type=float, default=0.20)
    args = parser.parse_args()
    assess(args.root, args.report, args.total_tokens, args.buffer)
