"""Migrate a completed pretraining checkpoint to a smaller DDP world.

This is an explicit, one-time migration for the GPU-3 retirement.  Model,
optimizer, scaler, and step metadata are retained.  Rank-local RNG and loader
state are rebuilt for the new world size, so the result is resumable but not a
bit-exact continuation of the old four-rank data order.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

# Permit the documented ``python train/migrate_checkpoint.py`` invocation:
# the script directory otherwise shadows the top-level ``train`` package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import DEFAULT_CONFIG
from train.data.loader import MultiSourceDataLoader


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _save(value: Any, path: Path) -> None:
    with path.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _config_digest() -> str:
    payload = dataclasses.asdict(DEFAULT_CONFIG)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _split(total: int, world_size: int, rank: int) -> int:
    base, extra = divmod(int(total), int(world_size))
    return base + (1 if rank < extra else 0)


def _advance(stream: Any, count: int) -> Dict[str, int]:
    """Return a valid SourceStream cursor after consuming ``count`` tokens."""
    remaining = int(count)
    for index, (_path, start, end) in enumerate(stream.segments):
        length = int(end) - int(start)
        if remaining < length:
            return {"shard_idx": index, "offset": int(start) + remaining}
        remaining -= length
        if remaining == 0:
            if index + 1 < len(stream.segments):
                return {
                    "shard_idx": index + 1,
                    "offset": int(stream.segments[index + 1][1]),
                }
            raise ValueError(f"cursor reaches the end of source {stream.name!r}")
    raise ValueError(
        f"source {stream.name!r} has insufficient tokens for migrated cursor "
        f"(remaining={remaining})"
    )


def _new_loader_state(
    loader: MultiSourceDataLoader,
    aggregate_counts: Dict[str, int],
    aggregate_mix: Dict[str, int],
    world_size: int,
    rank: int,
    target_weights: Dict[str, float],
) -> Dict[str, Any]:
    state = loader.state_dict()
    state["streams"] = {
        name: _advance(stream, _split(aggregate_counts.get(name, 0), world_size, rank))
        for name, stream in loader.streams.items()
    }
    state["token_counts"] = {
        name: _split(aggregate_counts.get(name, 0), world_size, rank)
        for name in loader.streams
    }
    state["mix_tokens_served"] = {
        name: _split(aggregate_mix.get(name, 0), world_size, rank)
        for name in loader.streams
    }
    state["total_tokens_served"] = sum(state["token_counts"].values())
    state["target_weights"] = dict(target_weights)
    return state


def migrate(
    checkpoint: Path,
    train_manifest: Path,
    validation_manifest: Path,
    new_world_size: int,
    archive_root: Path,
) -> Path:
    if new_world_size < 1:
        raise ValueError("new_world_size must be positive")
    if not (checkpoint / "COMPLETE").is_file():
        raise FileNotFoundError(f"source is not a COMPLETE checkpoint: {checkpoint}")
    if checkpoint.name != "step_004000":
        raise ValueError("this guarded migration is intended only for step_004000")

    old_meta = _load(checkpoint / "meta.pt")
    old_world_size = int(old_meta.get("world_size", 1))
    if old_world_size <= new_world_size:
        raise ValueError(f"expected a reduction from 4 ranks, got {old_world_size} -> {new_world_size}")
    old_states = []
    for rank in range(old_world_size):
        path = checkpoint / f"loader_rank{rank}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        old_states.append(_load(path))

    old_train = [state["train"] for state in old_states]
    target_weights = dict(old_train[0]["target_weights"])
    source_names = sorted(old_train[0]["streams"])
    aggregate_counts = {
        name: sum(int(state["token_counts"].get(name, 0)) for state in old_train)
        for name in source_names
    }
    aggregate_mix = {
        name: sum(int(state["mix_tokens_served"].get(name, 0)) for state in old_train)
        for name in source_names
    }

    train_loaders = [
        MultiSourceDataLoader(
            manifest_path=str(train_manifest),
            seq_len=DEFAULT_CONFIG.sequence_length,
            batch_size=DEFAULT_CONFIG.micro_batch_size,
            rank=rank,
            world_size=new_world_size,
        )
        for rank in range(new_world_size)
    ]
    val_loaders = [
        MultiSourceDataLoader(
            manifest_path=str(validation_manifest),
            seq_len=DEFAULT_CONFIG.sequence_length,
            batch_size=DEFAULT_CONFIG.micro_batch_size,
            rank=rank,
            world_size=new_world_size,
        )
        for rank in range(new_world_size)
    ]
    for loader in train_loaders:
        loader.set_mix(target_weights)

    migrated_loader_states = []
    for rank in range(new_world_size):
        migrated_loader_states.append(
            {
                "train": _new_loader_state(
                    train_loaders[rank], aggregate_counts, aggregate_mix,
                    new_world_size, rank, target_weights,
                ),
                "validation": val_loaders[rank].state_dict(),
            }
        )

    run_signature = dict(old_meta["run_signature"])
    run_signature["world_size"] = new_world_size
    run_signature["gradient_accumulation_steps"] = DEFAULT_CONFIG.gradient_accumulation_steps
    run_signature["config_sha256"] = _config_digest()
    new_meta = dict(old_meta)
    new_meta["world_size"] = new_world_size
    new_meta["run_signature"] = run_signature
    new_meta["migration"] = {
        "from_world_size": old_world_size,
        "to_world_size": new_world_size,
        "source_checkpoint": str(checkpoint),
        "data_cursor_policy": "aggregate_per_source_then_split_across_new_ranks",
        "bit_exact": False,
    }

    archive_root.mkdir(parents=True, exist_ok=False)
    archive = archive_root / f"{checkpoint.name}-from-{old_world_size}gpu"
    if archive.exists():
        raise FileExistsError(archive)
    tmp = checkpoint.parent / f".tmp_{checkpoint.name}_{new_world_size}gpu"
    if tmp.exists():
        raise FileExistsError(tmp)

    moved = False
    try:
        checkpoint.rename(archive)
        moved = True
        tmp.mkdir()
        for name in ("model.pt", "optimizer.pt", "scaler.pt"):
            os.link(archive / name, tmp / name)
        _save(new_meta, tmp / "meta.pt")
        for rank in range(new_world_size):
            os.link(archive / f"rng_rank{rank}.pt", tmp / f"rng_rank{rank}.pt")
            _save(migrated_loader_states[rank], tmp / f"loader_rank{rank}.pt")
        (tmp / "COMPLETE").write_text(
            f"step={new_meta['step']}\nworld_size={new_world_size}\n",
            encoding="utf-8",
        )
        with (tmp / "COMPLETE").open("rb") as handle:
            os.fsync(handle.fileno())
        tmp.rename(checkpoint)
        return checkpoint
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        if moved and not checkpoint.exists():
            archive.rename(checkpoint)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--new-world-size", type=int, default=3)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    result = migrate(
        args.checkpoint, args.train_manifest, args.validation_manifest,
        args.new_world_size, args.archive_root,
    )
    print(f"migrated checkpoint: {result}")


if __name__ == "__main__":
    main()
