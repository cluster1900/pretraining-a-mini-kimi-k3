"""CLI for the same fail-closed checks used by ``train/train.py``."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CANONICAL_PARAMETER_COUNTS, DEFAULT_CONFIG
from train.readiness import check_training_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Check audited data before pretraining")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--coverage", default=None)
    parser.add_argument("--allow-incomplete-pipeline", action="store_true")
    parser.add_argument("--expected-training-tokens", type=int, default=None)
    args = parser.parse_args()
    result = check_training_readiness(
        args.manifest,
        args.validation_manifest,
        vocab_size=DEFAULT_CONFIG.vocab_size,
        stable_mix=DEFAULT_CONFIG.stable_mix,
        require_pipeline_complete=not args.allow_incomplete_pipeline,
        coverage_path=args.coverage,
        expected_parameters=CANONICAL_PARAMETER_COUNTS["mini-k3"],
        expected_training_tokens=args.expected_training_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
