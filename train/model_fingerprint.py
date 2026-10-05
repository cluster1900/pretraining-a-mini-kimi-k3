"""Fingerprint of the model code a smoke run exercised.

``smoke_from_manifest.py`` records ``model_code_sha256`` in ``SMOKE.json`` and
``readiness.py`` refuses to start training when the current model code differs
(AGENTS.md rules 4 and 6): a smoke that passed on older model code says nothing
about the code that would now be trained.

Covered files: ``train/config.py``, ``train/models/*.py``, ``train/kernels/*.py``
and the engine files exercised by the smoke (``muon.py`` and ``balancer.py``).
The digest is over the sorted repository-relative paths and the exact file
bytes, so it is independent of the checkout location.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

TRAIN_DIR = Path(__file__).resolve().parent


def model_code_files(train_dir: Path | str = TRAIN_DIR) -> List[Path]:
    root = Path(train_dir)
    files = [
        root / "config.py",
        root / "engine" / "muon.py",
        root / "engine" / "balancer.py",
    ]
    files += list((root / "models").glob("*.py"))
    files += list((root / "kernels").glob("*.py"))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Model code files missing: {missing}")
    return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())


def _digest(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "little"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def model_code_fingerprint(train_dir: Path | str = TRAIN_DIR) -> str:
    """SHA256 over the sorted model code files (path + contents)."""
    root = Path(train_dir).resolve()
    return _digest(model_code_files(root), root)


if __name__ == "__main__":
    print(model_code_fingerprint())
