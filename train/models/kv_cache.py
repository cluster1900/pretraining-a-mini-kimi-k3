"""Inference cache for recurrent KDA and windowed MLA layers.

The cache is deliberately explicit and serializable so rollout workers cannot
accidentally reuse state from another request.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch

@dataclass
class KVCache:
    position: int = 0
    kda_states: List[Optional[torch.Tensor]] = field(default_factory=list)
    kda_conv_states: List[Optional[Any]] = field(default_factory=list)
    mla_keys: List[Optional[torch.Tensor]] = field(default_factory=list)
    mla_values: List[Optional[torch.Tensor]] = field(default_factory=list)

    def reset(self):
        self.position = 0
        self.kda_states = [None for _ in self.kda_states]
        self.kda_conv_states = [None for _ in self.kda_conv_states]
        self.mla_keys = [None for _ in self.mla_keys]
        self.mla_values = [None for _ in self.mla_values]

    def detach(self):
        self.kda_states = [x.detach() if x is not None else None for x in self.kda_states]
        self.kda_conv_states = [
            tuple(t.detach() for t in x) if x is not None else None
            for x in self.kda_conv_states
        ]
        self.mla_keys = [x.detach() if x is not None else None for x in self.mla_keys]
        self.mla_values = [x.detach() if x is not None else None for x in self.mla_values]

    def to(self, device):
        for name in ("kda_states", "mla_keys", "mla_values"):
            setattr(self, name, [x.to(device) if x is not None else None for x in getattr(self, name)])
        self.kda_conv_states = [
            tuple(t.to(device) for t in x) if x is not None else None
            for x in self.kda_conv_states
        ]
        return self
