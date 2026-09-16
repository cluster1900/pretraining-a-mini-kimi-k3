"""
No-Auxiliary-Loss (noaux_tc) MoE Load Balancer.
Updates per-expert selection bias without contaminating the task loss function.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Tuple


class NoAuxBalancer:
    """
    Manages the e_score_correction_bias for all MoE layers in the model.
    Rule:
        bias_i <- bias_i + gamma * sign(mean_load - load_i)
    """
    def __init__(self, model: nn.Module, gamma: float = 1e-2):
        self.model = model
        self.gamma = gamma
        self.moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, "is_moe_layer") and module.is_moe_layer:
                self.moe_layers.append(module)

    @torch.no_grad()
    def step(self) -> Dict[str, float]:
        """
        Applies sign-based bias adjustment to all MoE layers.
        Returns global telemetry statistics:
        - dead_frac: fraction of experts receiving 0 tokens in this step
        - imbalance: max_load / mean_load of expert utilization
        """
        if not self.moe_layers:
            return {"dead_frac": 0.0, "imbalance": 1.0}
        
        total_dead = 0
        total_experts = 0
        max_imbalance = 1.0
        
        for layer in self.moe_layers:
            gate = layer.gate
            load = gate.expert_load.float()  # [num_experts]
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(load, op=dist.ReduceOp.SUM)
            total_tokens = load.sum()
            
            if total_tokens > 0:
                mean_load = load.mean()
                diff = mean_load - load
                # Sign update: push underloaded experts up, overloaded experts down
                gate.e_score_correction_bias.add_(torch.sign(diff) * self.gamma)
                
                # Compute telemetry metrics
                dead_count = (load == 0).sum().item()
                total_dead += dead_count
                total_experts += len(load)
                
                if mean_load > 0:
                    imbalance = (load.max() / mean_load).item()
                    if imbalance > max_imbalance:
                        max_imbalance = imbalance
            
            # Reset accumulator for next forward step
            gate.expert_load.zero_()
            
        dead_frac = total_dead / max(1, total_experts)
        return {
            "dead_frac": dead_frac,
            "imbalance": max_imbalance,
        }
