"""pytest configuration for train/ (lets plain `pytest train/data` work from the repo root).

- Puts the repo root and train/data on sys.path, matching the documented
  `PYTHONPATH=train/data python -m unittest discover -s train/data -p 'test_*.py'`
  setup: data tests use flat imports (`from canonical_v2 import ...`) and
  package imports (`from train.config import ...`).
- Provides the `device` fixture that the script-style checks in
  test_cache_equivalence.py take as an argument; `python train/test_cache_equivalence.py`
  is unaffected and still picks the device in main().
"""

import sys
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).resolve().parent
for path in (TRAIN_DIR.parent, TRAIN_DIR / "data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
