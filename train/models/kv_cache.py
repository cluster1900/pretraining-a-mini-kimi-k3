"""Inference cache for recurrent KDA and CSA2 MLA layers.

The cache is deliberately explicit and serializable so rollout workers cannot
accidentally reuse state from another request.

MLA slots (``mla_keys[i]``) hold ``{"tail": raw latents, "entries": EntryStore
or None, "count": complete groups}``. Only encoder (``full``) MLA layers own an
EntryStore; reindex/reuse layers read the encoder's entries.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch

from train.models.fp4 import PackedKV, dequantize_fp4, quantize_fp4


class EntryStore:
    """Append-only [B, count, rank] buffer with capacity doubling.

    ``fp4=True`` stores E2M1 codes plus one FP16 scale per entry (opt-in,
    approximate). ``dense()`` returns the live entries as one tensor.
    """

    def __init__(self, fp4: bool = False):
        self.fp4 = fp4
        self.buffers: Optional[List[torch.Tensor]] = None
        self.count = 0

    def _parts(self, values: torch.Tensor):
        if self.fp4:
            blob = quantize_fp4(values)
            return [blob.packed, blob.scale]
        return [values]

    def append(self, values: torch.Tensor) -> None:
        added = values.shape[1]
        if added == 0:
            return
        parts = self._parts(values.detach())
        needed = self.count + added
        if self.buffers is None:
            capacity = max(needed, 64)
            self.buffers = [p.new_empty(p.shape[0], capacity, *p.shape[2:]) for p in parts]
        elif needed > self.buffers[0].shape[1]:
            capacity = max(needed, 2 * self.buffers[0].shape[1])
            grown = []
            for old in self.buffers:
                new = old.new_empty(old.shape[0], capacity, *old.shape[2:])
                new[:, : self.count] = old[:, : self.count]
                grown.append(new)
            self.buffers = grown
        for buffer, part in zip(self.buffers, parts):
            buffer[:, self.count:needed] = part
        self.count = needed

    def dense(self) -> Optional[torch.Tensor]:
        if not self.count:
            return None
        if self.fp4:
            packed, scale = self.buffers
            return dequantize_fp4(PackedKV(packed[:, : self.count], scale[:, : self.count]))
        return self.buffers[0][:, : self.count]

    @property
    def shape(self):
        if not self.count:
            return torch.Size((0, 0, 0))
        dense_last = self.buffers[0].shape[-1] * (2 if self.fp4 else 1)
        return torch.Size((self.buffers[0].shape[0], self.count, dense_last))

    def _rebuilt(self, fn):
        copy = EntryStore(self.fp4)
        copy.count = self.count
        copy.buffers = None if self.buffers is None else [fn(b[:, : self.count]) for b in self.buffers]
        return copy

    def clone(self):
        return self._rebuilt(lambda t: t.detach().clone())

    def detach(self):
        return self._rebuilt(lambda t: t.detach())

    def to(self, device):
        return self._rebuilt(lambda t: t.to(device))


def _map_cached(value, tensor_fn, object_fn):
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_map_cached(item, tensor_fn, object_fn) for item in value)
    if isinstance(value, list):
        return [_map_cached(item, tensor_fn, object_fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_cached(item, tensor_fn, object_fn) for key, item in value.items()}
    if torch.is_tensor(value):
        return tensor_fn(value)
    return object_fn(value)


def copy_cached(value):
    return _map_cached(value, lambda t: t.detach().clone(), lambda o: o.clone())


def move_cached(value, device):
    return _map_cached(value, lambda t: t.to(device), lambda o: o.to(device))


def detach_cached(value):
    return _map_cached(value, lambda t: t.detach(), lambda o: o.detach())


@dataclass
class KVCache:
    position: int = 0
    kda_states: List[Optional[torch.Tensor]] = field(default_factory=list)
    kda_conv_states: List[Optional[Any]] = field(default_factory=list)
    mla_keys: List[Optional[Any]] = field(default_factory=list)
    mla_values: List[Optional[torch.Tensor]] = field(default_factory=list)  # unused; kept for layout compatibility
    token_ids: Optional[torch.Tensor] = None

    def clone(self):
        """Copy cache tensors so a rejected speculative token can be rolled back."""
        copied = KVCache(position=self.position)
        copied.kda_states = [copy_cached(value) for value in self.kda_states]
        copied.kda_conv_states = [copy_cached(value) for value in self.kda_conv_states]
        copied.mla_keys = [copy_cached(value) for value in self.mla_keys]
        copied.mla_values = [copy_cached(value) for value in self.mla_values]
        copied.token_ids = copy_cached(self.token_ids)
        return copied

    def reset(self):
        self.position = 0
        self.kda_states = [None for _ in self.kda_states]
        self.kda_conv_states = [None for _ in self.kda_conv_states]
        self.mla_keys = [None for _ in self.mla_keys]
        self.mla_values = [None for _ in self.mla_values]
        self.token_ids = None

    def detach(self):
        self.kda_states = [detach_cached(x) for x in self.kda_states]
        self.kda_conv_states = [detach_cached(x) for x in self.kda_conv_states]
        self.mla_keys = [detach_cached(x) for x in self.mla_keys]
        self.mla_values = [detach_cached(x) for x in self.mla_values]
        self.token_ids = detach_cached(self.token_ids)

    def to(self, device):
        self.kda_states = [move_cached(x, device) for x in self.kda_states]
        self.kda_conv_states = [move_cached(x, device) for x in self.kda_conv_states]
        self.mla_keys = [move_cached(x, device) for x in self.mla_keys]
        self.mla_values = [move_cached(x, device) for x in self.mla_values]
        self.token_ids = move_cached(self.token_ids, device)
        return self
