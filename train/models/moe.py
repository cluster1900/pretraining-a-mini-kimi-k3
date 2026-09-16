"""
Kimi MoE Block with TrainableMoEGate and Differentiable Dispatch.
Implements:
- Sigmoid Router with noaux_tc bias correction
- Latent MoE bottleneck (hidden -> hidden // 2)
- 256 routed experts using FusedSitu activation
- 2 shared experts in full hidden width
- Differentiable batch MoE dispatch (eliminating official NotImplementedError)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from train.config import MiniK3Config
from train.kernels.situ_fused import FusedSitu


class ExpertMLP(nn.Module):
    """Single expert MLP with situ activation."""
    def __init__(self, in_features: int, intermediate_features: int, beta: float = 4.0, linear_beta: float = 25.0):
        super().__init__()
        # Gate and up projection combined: [in_features -> 2 * intermediate]
        self.gate_up_proj = nn.Linear(in_features, 2 * intermediate_features, bias=False)
        self.down_proj = nn.Linear(intermediate_features, in_features, bias=False)
        self.act = FusedSitu(beta=beta, linear_beta=linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up_proj(x)
        h = self.act(gu)
        return self.down_proj(h)


class TrainableMoEGate(nn.Module):
    """
    Trainable Sigmoid Gate with noaux_tc load correction bias.
    Fixes the official `assert not self.training` and uninitialized bias bugs.
    """
    def __init__(self, config: MiniK3Config):
        super().__init__()
        self.num_experts = config.num_routed_experts
        self.top_k = config.top_k
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)

        # e_score_correction_bias is updated by NoAuxBalancer, never by AdamW
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer("expert_load", torch.zeros(self.num_experts, dtype=torch.long), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [tokens, hidden_size]
        # 1. Compute scores with sigmoid in float32
        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        scores = torch.sigmoid(logits)  # [tokens, num_experts]

        # 2. Add unbiased bias for top-k selection ONLY
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)

        # 3. Weights come from the ORIGINAL, UNBIASED scores
        topk_weights = scores.gather(dim=-1, index=topk_idx)
        # Re-normalize weights to sum to 1.0 across selected top_k
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # 4. Track expert token count for load balancing
        if self.training:
            flat_indices = topk_idx.reshape(-1)
            counts = torch.bincount(flat_indices, minlength=self.num_experts)
            self.expert_load.add_(counts)

        return topk_idx, topk_weights.to(hidden_states.dtype)


class KimiMoEBlock(nn.Module):
    def __init__(self, config: MiniK3Config):
        super().__init__()
        self.is_moe_layer = True
        self.config = config
        self.hidden_size = config.hidden_size
        self.latent_dim = config.routed_expert_hidden_size
        self.num_experts = config.num_routed_experts
        self.top_k = config.top_k

        # Router Gate
        self.gate = TrainableMoEGate(config)

        # Latent MoE down/up projections
        self.latent_down = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.latent_up = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.latent_norm = nn.RMSNorm(self.latent_dim, eps=config.rms_norm_eps)

        # 256 Routed Experts
        self.experts = nn.ModuleList([
            ExpertMLP(
                in_features=self.latent_dim,
                intermediate_features=config.moe_intermediate_size,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )
            for _ in range(self.num_experts)
        ])

        # 2 Shared Experts (running in full hidden width)
        shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
        self.shared_experts = ExpertMLP(
            in_features=self.hidden_size,
            intermediate_features=shared_intermediate,
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, hidden_size]
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        n_tok = x_flat.shape[0]

        # 1. Unconditional Shared Expert path
        shared_out = self.shared_experts(x_flat)

        # 2. Project to Latent Space for routed experts
        x_lat = self.latent_norm(self.latent_down(x_flat))  # [tokens, latent_dim]

        # 3. Route tokens
        topk_idx, topk_weight = self.gate(x_flat)  # [tokens, k], [tokens, k]

        # 4. Differentiable MoE Dispatch
        flat_idx = topk_idx.reshape(-1)            # [tokens * k]
        order = flat_idx.argsort()
        sorted_tokens = x_lat[order // self.top_k]  # Gather sorted tokens
        counts = torch.bincount(flat_idx, minlength=self.num_experts)

        chunks = []
        start = 0
        counts_list = counts.tolist()
        for i, c in enumerate(counts_list):
            if c == 0:
                continue
            expert_in = sorted_tokens[start : start + c]
            chunks.append(self.experts[i](expert_in))
            start += c

        if chunks:
            dispatched_out = torch.cat(chunks, dim=0)
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), device=order.device)
            restored_tokens = dispatched_out[inv_order].view(n_tok, self.top_k, self.latent_dim)
            # Weighted combine
            routed_lat = (restored_tokens * topk_weight.unsqueeze(-1)).sum(dim=1)
        else:
            routed_lat = torch.zeros_like(x_lat)

        # 5. Project Latent back to Hidden & combine with Shared Experts
        routed_out = self.latent_up(routed_lat)
        out = routed_out + shared_out
        return out.view(*orig_shape)
