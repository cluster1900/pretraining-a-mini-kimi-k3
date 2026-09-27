"""
Kimi MoE Block with TrainableMoEGate and Differentiable Dispatch.
Implements:
- Sigmoid router. Quantile balancing uses the bias only for expert choice.
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
    Trainable sigmoid gate. The bias changes which experts are chosen, not the mixture weights.
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
        self.hist_bins = 256
        self.hist_lo = -2.0
        self.hist_hi = 2.0
        self.register_buffer(
            "margin_hist",
            torch.zeros(self.num_experts, self.hist_bins, dtype=torch.float32),
            persistent=False,
        )
        self._last_counts = None
        self._last_scores = None
        self._last_alpha = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [tokens, hidden_size]
        # 1. Compute scores with sigmoid in float32
        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        scores = torch.sigmoid(logits)  # [tokens, num_experts]

        # 2. Bias corrects selection only. The extra candidate is the cutoff, not a route.
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        chosen, topk_all = torch.topk(scores_for_choice, k=self.top_k + 1, dim=-1, sorted=True)
        topk_idx = topk_all[:, : self.top_k]
        alpha = chosen[:, self.top_k]

        # 3. Weights come from the ORIGINAL, UNBIASED scores
        topk_weights = scores.gather(dim=-1, index=topk_idx)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # 4. The layer copies these once, outside activation recomputation.
        if self.training:
            flat_indices = topk_idx.reshape(-1)
            counts = torch.bincount(flat_indices, minlength=self.num_experts)
            self._last_counts = counts.detach()
            self._last_scores = scores.detach()
            self._last_alpha = alpha.detach()

        return topk_idx, topk_weights.to(hidden_states.dtype)

    def accumulate_margin_histogram(self) -> None:
        """Bin score-minus-cutoff margins. Called once per forward, not on recompute."""
        if self._last_scores is None or self._last_alpha is None:
            return
        margins = self._last_scores.float() - self._last_alpha.float().unsqueeze(-1)
        bins = self.margin_hist.shape[-1]
        span = self.hist_hi - self.hist_lo
        scaled = (margins.clamp(self.hist_lo, self.hist_hi) - self.hist_lo) / span
        idx = (scaled * (bins - 1)).round().long().clamp(0, bins - 1)
        tokens, experts = idx.shape
        offsets = torch.arange(experts, device=idx.device) * bins
        flat = idx.transpose(0, 1).reshape(-1) + offsets.repeat_interleave(tokens)
        counts = torch.bincount(flat, minlength=experts * bins).view(experts, bins).to(self.margin_hist.dtype)
        self.margin_hist += counts
        self._last_scores = None
        self._last_alpha = None


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

        # Two always-on experts at full hidden width. Their outputs are summed.
        self.shared_experts = nn.ModuleList([
            ExpertMLP(
                in_features=self.hidden_size,
                intermediate_features=config.moe_intermediate_size,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )
            for _ in range(config.num_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, hidden_size]
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        n_tok = x_flat.shape[0]

        # 1. Unconditional shared experts
        shared_out = self.shared_experts[0](x_flat)
        for expert in self.shared_experts[1:]:
            shared_out = shared_out + expert(x_flat)

        # 2. Routed experts run in the latent. Normalize after they are mixed.
        x_lat = self.latent_down(x_flat)

        # 3. Route tokens
        topk_idx, topk_weight = self.gate(x_flat)  # [tokens, k], [tokens, k]

        # 4. One batched expert GEMM. Counts stay on device; no per-expert Python launch.
        flat_idx = topk_idx.reshape(-1)
        order = flat_idx.argsort()
        sorted_tokens = x_lat[order // self.top_k]
        counts = torch.bincount(flat_idx, minlength=self.num_experts)
        max_count = int(counts.max().item()) if flat_idx.numel() else 0
        if max_count == 0:
            routed_lat = torch.zeros_like(x_lat)
        else:
            dispatched = self._batched_experts(sorted_tokens, counts, max_count)
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), device=order.device)
            restored_tokens = dispatched[inv_order].view(n_tok, self.top_k, self.latent_dim)
            routed_lat = (restored_tokens * topk_weight.unsqueeze(-1)).sum(dim=1)

        # 5. RMSNorm sits between the mixture and the up projection.
        routed_out = self.latent_up(self.latent_norm(routed_lat))
        out = routed_out + shared_out
        return out.view(*orig_shape)

    def _batched_experts(self, sorted_tokens, counts, max_count):
        """Run every routed expert as one [E, capacity, D] GEMM."""
        device = sorted_tokens.device
        positions = torch.arange(sorted_tokens.shape[0], device=device)
        offsets = torch.cumsum(counts, 0)
        expert_ids = torch.searchsorted(offsets, positions, right=True)
        previous = torch.cat([offsets.new_zeros(1), offsets[:-1]])
        local = positions - previous[expert_ids]
        packed = sorted_tokens.new_zeros(self.num_experts, max_count, self.latent_dim)
        packed[expert_ids, local] = sorted_tokens
        gate_w = torch.stack([expert.gate_up_proj.weight for expert in self.experts], dim=0)
        down_w = torch.stack([expert.down_proj.weight for expert in self.experts], dim=0)
        hidden = torch.bmm(packed, gate_w.transpose(1, 2))
        hidden = self.experts[0].act(hidden)
        out = torch.bmm(hidden, down_w.transpose(1, 2))
        return out[expert_ids, local]
