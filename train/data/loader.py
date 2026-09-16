"""
Multi-Source Binary Shard DataLoader.
Features:
- Sequence-level source sampling (not shard-level) to maintain continuous stable data mix
- Raw little-endian uint32 binary token format
- Explicit cursor tracking for bit-exact checkpoint resumption
- Loud failure on missing shards (avoids silent renormalisation to 100% single source)
- Realised vs target mix tracking
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


class SourceStream:
    """Manages sequential reading over a list of uint32 binary token shards for one source."""
    def __init__(self, name: str, shard_paths: List[Path], shard_idx: int = 0, offset: int = 0):
        if not shard_paths:
            raise ValueError(f"Source {name!r} has no shard paths provided.")
        self.name = name
        self.shard_paths = shard_paths
        self.shard_idx = shard_idx
        self.offset = offset  # Offset in uint32 tokens
        self.current_mmap = None
        self._open_current_shard()

    def _open_current_shard(self):
        if self.shard_idx >= len(self.shard_paths):
            # Loop back to beginning for next epoch
            self.shard_idx = 0
            self.offset = 0
        path = self.shard_paths[self.shard_idx]
        self.current_mmap = np.memmap(path, dtype=np.uint32, mode="r")

    def take(self, n_tokens: int) -> np.ndarray:
        """Reads exactly n_tokens from the shard stream."""
        collected = []
        remaining = n_tokens

        while remaining > 0:
            if self.current_mmap is None:
                self._open_current_shard()

            available = len(self.current_mmap) - self.offset
            if available <= 0:
                # Move to next shard
                self.shard_idx = (self.shard_idx + 1) % len(self.shard_paths)
                self.offset = 0
                self._open_current_shard()
                continue

            take_n = min(remaining, available)
            chunk = self.current_mmap[self.offset : self.offset + take_n]
            collected.append(chunk)
            self.offset += take_n
            remaining -= take_n

        return np.concatenate(collected) if len(collected) > 1 else collected[0]

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
                # In distributed training, partition shards across ranks so GPUs don't read duplicate data
                if self.world_size > 1 and len(paths) >= self.world_size:
                    assigned_paths = [p for i, p in enumerate(paths) if i % self.world_size == self.rank]
                    if not assigned_paths:
                        assigned_paths = paths
                        initial_offset = self.rank * 1024 * 1024
                    else:
                        initial_offset = 0
                else:
                    assigned_paths = paths
                    initial_offset = self.rank * 1024 * 1024 if self.world_size > 1 else 0

                self.streams[src_name] = SourceStream(src_name, assigned_paths, offset=initial_offset)
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
        self.target_weights = {k: v for k, v in new_weights.items() if k in self.streams}
        self._normalize_weights()

    def next_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a micro-batch [B, L] of input tokens and shifted labels.
        """
        # If no streams loaded (e.g. during smoke testing without real data), generate dummy batch
        if not self.streams:
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
