"""
Quantile Balancing for auxiliary-loss-free MoE routing.

The selection bias is the centered negative quantile of (router score - cutoff).
It is applied on the next step, never inside the batch that produced it, and
AdamW does not see it. `gamma` remains in the constructor so older call sites
keep working; the quantile update does not use a fixed step size.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict


def bias_from_histogram(hist: torch.Tensor, level: float, lo: float, hi: float) -> torch.Tensor:
    """Read one quantile per expert from pooled bin counts and center the result."""
    total = hist.sum(dim=-1).clamp(min=1)
    cdf = torch.cumsum(hist, dim=-1)
    target = level * total
    reached = cdf >= target.unsqueeze(-1)
    bin_idx = reached.to(torch.int64).argmax(dim=-1)
    width = (hi - lo) / hist.shape[-1]
    value = lo + (bin_idx.float() + 0.5) * width
    bias = -value
    return bias - bias.mean()


class NoAuxBalancer:
    """
    Updates e_score_correction_bias on every MoE layer.
    Telemetry stays dead_frac and imbalance. Router-score entropy is not used.
    """
    def __init__(self, model: nn.Module, gamma: float = 1e-2):
        self.model = model
        self.gamma = gamma
        self.moe_layers = []
        for module in model.modules():
            if hasattr(module, "is_moe_layer") and module.is_moe_layer:
                self.moe_layers.append(module)

    @torch.no_grad()
    def step(self) -> Dict[str, float]:
        if not self.moe_layers:
            return {"dead_frac": 0.0, "imbalance": 1.0}

        total_dead = 0
        total_experts = 0
        max_imbalance = 1.0

        for layer in self.moe_layers:
            gate = layer.gate
            load = gate.expert_load.float()
            hist = gate.margin_hist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(load, op=dist.ReduceOp.SUM)
                dist.all_reduce(hist, op=dist.ReduceOp.SUM)
            total_tokens = load.sum()

            if total_tokens > 0:
                mean_load = load.mean()
                dead_count = (load == 0).sum().item()
                total_dead += dead_count
                total_experts += len(load)
                if mean_load > 0:
                    imbalance = (load.max() / mean_load).item()
                    if imbalance > max_imbalance:
                        max_imbalance = imbalance

            if float(hist.sum()) > 0:
                level = 1.0 - (gate.top_k / gate.num_experts)
                bias = bias_from_histogram(hist, level, gate.hist_lo, gate.hist_hi)
                gate.e_score_correction_bias.copy_(bias.to(gate.e_score_correction_bias.dtype))

            gate.expert_load.zero_()
            gate.margin_hist.zero_()

        dead_frac = total_dead / max(1, total_experts)
        return {
            "dead_frac": dead_frac,
            "imbalance": max_imbalance,
        }
