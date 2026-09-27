"""Inference cache for recurrent KDA and windowed MLA layers.

The cache is deliberately explicit and serializable so rollout workers cannot
accidentally reuse state from another request.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch

def copy_cached(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(copy_cached(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().clone()
    return value.clone()


def move_cached(value, device):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(move_cached(item, device) for item in value)
    return value.to(device)


def detach_cached(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(detach_cached(item) for item in value)
    if torch.is_tensor(value):
        return value.detach()
    return value.detach()


@dataclass
class KVCache:
    position: int = 0
    kda_states: List[Optional[torch.Tensor]] = field(default_factory=list)
    kda_conv_states: List[Optional[Any]] = field(default_factory=list)
    mla_keys: List[Optional[torch.Tensor]] = field(default_factory=list)
    mla_values: List[Optional[torch.Tensor]] = field(default_factory=list)
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
