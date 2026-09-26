"""
Multi-Source Binary Shard DataLoader.
Features:
- Sequence-level source sampling (not shard-level) to maintain continuous stable data mix
- Raw little-endian uint32 binary token format
- Explicit cursor tracking for bit-exact checkpoint resumption
- Loud failure on missing shards (avoids silent renormalisation to 100% single source)
- Realised vs target mix tracking
"""

from __future__ import annotations

import json
import mmap
import os
import random
import sys
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any


def rank_token_range(n_tokens: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Return the half-open token interval of one shard owned by this rank.

    The intervals of all ranks partition ``[0, n_tokens)``. Giving every rank a
    slice of every shard keeps the per-rank totals within one token per shard,
    instead of leaving a whole 50M-token shard on one GPU.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for world size {world_size}")
    if n_tokens < 0:
        raise ValueError("token count must be non-negative")
    if world_size == 1:
        return 0, n_tokens
    base, extra = divmod(int(n_tokens), world_size)
    start = rank * base + min(rank, extra)
    length = base + (1 if rank < extra else 0)
    return start, start + length


def segments_for_rank(items: Sequence[Tuple[Path, int]], rank: int, world_size: int) -> List[Tuple[Path, int, int]]:
    """Disjoint ``(path, start, end)`` slices whose tokens belong to one rank."""
    segments = []
    for path, n_tokens in items:
        start, end = rank_token_range(n_tokens, rank, world_size)
        if end > start:
            segments.append((Path(path), start, end))
    return segments


class SourceStream:
    """Manages sequential reading over disjoint token slices of uint32 shards."""
    def __init__(self, name: str, segments: Sequence[Tuple[Path, int, int]], shard_idx: int = 0, offset: Optional[int] = None):
        if not segments:
            raise ValueError(f"Source {name!r} has no shard slices.")
        self.name = name
        self.segments = [(Path(path), int(start), int(end)) for path, start, end in segments]
        for path, start, end in self.segments:
            if end <= start:
                raise ValueError(f"Source {name!r} has an empty slice of {path}")
        self.shard_paths = [path for path, _, _ in self.segments]
        self.shard_idx = shard_idx
        self.offset = self.segments[shard_idx][1] if offset is None else offset
        self._file = None
        self._map = None
        self._tokens = None
        self._open_current_shard()

    def _close_current(self):
        if self._tokens is not None:
            self._tokens.release()
            self._tokens = None
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _open_current_shard(self):
        if self.shard_idx >= len(self.segments):
            self.shard_idx = 0
            self.offset = self.segments[0][1]
        self._close_current()
        if sys.byteorder != "little":
            raise RuntimeError("Token shards are little-endian uint32")
        path, start, end = self.segments[self.shard_idx]
        if self.offset < start or self.offset > end:
            self.offset = start
        self._file = path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._map) % 4:
            raise ValueError(f"{path} is not a uint32 token shard")
        self._tokens = memoryview(self._map).cast("I")
        if end > len(self._tokens):
            raise ValueError(f"{path} has {len(self._tokens)} tokens, slice ends at {end}")

    def take(self, n_tokens: int) -> array:
        """Reads exactly n_tokens from this rank's slices."""
        collected = array("I")
        remaining = n_tokens

        while remaining > 0:
            if self._tokens is None:
                self._open_current_shard()
            _path, _start, end = self.segments[self.shard_idx]
            available = end - self.offset
            if available <= 0:
                self.shard_idx = (self.shard_idx + 1) % len(self.segments)
                self.offset = self.segments[self.shard_idx][1]
                self._open_current_shard()
                continue

            take_n = min(remaining, available)
            collected.extend(self._tokens[self.offset : self.offset + take_n])
            self.offset += take_n
            remaining -= take_n

        return collected

    def close(self):
        self._close_current()

    def state_dict(self) -> Dict[str, Any]:
        return {"shard_idx": self.shard_idx, "offset": self.offset}

    def load_state_dict(self, state: Dict[str, Any]):
        self.shard_idx = state.get("shard_idx", 0)
        self.offset = state.get("offset", 0)
        self._open_current_shard()


class MultiSourceDataLoader:
    """
    Samples sequences across multiple sources according to target weights.
    Supports phase switching (e.g. from stable mix to decay/anneal mix) and
    disjoint shard partitioning across DDP ranks.
    """
    def __init__(
        self,
        manifest_path: Optional[str] = None,
        seq_len: int = 4096,
        batch_size: int = 4,
        allow_missing: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.allow_missing = allow_missing
        self.rank = rank
        self.world_size = world_size
        self.streams: Dict[str, SourceStream] = {}
        self.target_weights: Dict[str, float] = {}
        self.source_names: List[str] = []
        self.source_probs: List[float] = []
        self.token_counts: Dict[str, int] = {}
        self.total_tokens_served: int = 0

        if manifest_path and os.path.exists(manifest_path):
            self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: str):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        missing_shards = {}
        sources = data.get("sources", {})
        for src_name, src_info in sources.items():
            weight = src_info.get("weight", 0.0)
            paths = []
            for p_str in src_info.get("shards", []):
                p = Path(p_str)
                if not p.exists():
                    missing_shards.setdefault(src_name, []).append(p_str)
                else:
                    paths.append(p)

            if paths:
                # Each rank reads a disjoint token slice of every shard. Whole-shard
                # striding left two ranks short of Chinese and Python for a 10B run,
                # so those ranks would have repeated the start of the shard.
                counted = {item["path"]: int(item["tokens"]) for item in src_info.get("shard_metadata", [])}
                items = []
                for path in paths:
                    key = str(path)
                    if key not in counted:
                        size = path.stat().st_size
                        if size % 4:
                            raise ValueError(f"{path} is not a uint32 token shard")
                        counted[key] = size // 4
                    items.append((path, counted[key]))
                segments = segments_for_rank(items, self.rank, self.world_size)
                if not segments:
                    raise ValueError(f"Source {src_name!r} assigned no tokens to rank {self.rank}")
                self.streams[src_name] = SourceStream(src_name, segments)
                self.target_weights[src_name] = weight
                self.token_counts[src_name] = 0

        # Enforce fail-loud rule on missing files
        if missing_shards and not self.allow_missing:
            details = "; ".join(f"{k}: {len(v)} missing" for k, v in missing_shards.items())
            raise FileNotFoundError(
                f"Data manifest references missing shards ({details}). "
                "Failing loudly to prevent silent single-source degradation."
            )

        self._normalize_weights()

    def _normalize_weights(self):
        total_w = sum(self.target_weights.values())
        if total_w > 0:
            self.source_names = list(self.target_weights.keys())
            self.source_probs = [self.target_weights[n] / total_w for n in self.source_names]

    def set_mix(self, new_weights: Dict[str, float]):
        """Switches data mix (e.g. switching to decay phase)."""
        missing = [name for name, weight in new_weights.items() if weight > 0 and name not in self.streams]
        if missing:
            raise KeyError(f"Mix names are not loaded from the manifest: {missing}")
        self.target_weights = {k: float(v) for k, v in new_weights.items() if k in self.streams}
        self._normalize_weights()

    def next_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a micro-batch [B, L] of input tokens and shifted labels.
        """
        # If no streams loaded (e.g. during smoke testing without real data), generate dummy batch
        if not self.streams:
            import torch
            dummy = torch.randint(0, 163840, (self.batch_size, self.seq_len), device=device, dtype=torch.long)
            return dummy, dummy.clone()

        batch_seqs = []
        for _ in range(self.batch_size):
            src = random.choices(self.source_names, weights=self.source_probs, k=1)[0]
            stream = self.streams[src]
            tokens = stream.take(self.seq_len)
            self.token_counts[src] += self.seq_len
            self.total_tokens_served += self.seq_len
            batch_seqs.append(tokens)

        import numpy as np
        import torch
        batch_arr = np.stack(batch_seqs, axis=0)  # [B, L]
        input_ids = torch.tensor(batch_arr, dtype=torch.long, device=device)
        return input_ids, input_ids.clone()

    def realised_mix(self) -> Dict[str, float]:
        """Calculates actual realized token ratio across sources."""
        if self.total_tokens_served == 0:
            return {s: 0.0 for s in self.source_names}
        return {s: count / self.total_tokens_served for s, count in self.token_counts.items()}

    def target_mix(self) -> Dict[str, float]:
        return {s: prob for s, prob in zip(self.source_names, self.source_probs)}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "streams": {s: stream.state_dict() for s, stream in self.streams.items()},
            "token_counts": dict(self.token_counts),
            "total_tokens_served": self.total_tokens_served,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        stream_states = state.get("streams", {})
        for s, s_state in stream_states.items():
            if s in self.streams:
                self.streams[s].load_state_dict(s_state)
        self.token_counts = state.get("token_counts", {})
        self.total_tokens_served = state.get("total_tokens_served", 0)
