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
        # 1. Scores in float32. Autocast would otherwise run F.linear in FP16
        #    regardless of the .float() casts.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
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
            counts = torch.zeros(self.num_experts, dtype=torch.long, device=flat_indices.device)
            counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
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
        flat = (idx + offsets.view(1, -1)).reshape(-1)
        counts = torch.zeros(experts * bins, dtype=self.margin_hist.dtype, device=idx.device)
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        self.margin_hist += counts.view(experts, bins)
        self._last_scores = None
        self._last_alpha = None


class RoutedExperts(nn.Module):
    """All routed experts as two stacked weights, dispatched in fixed-size blocks.

    The earlier layout padded every expert to the busiest expert's token count.
    One 2048-token document can send ~900 of its 12,288 routings to a single
    expert, so 256 experts were padded to ~900 rows each and the situ
    temporaries alone exceeded the V100 headroom.  Here each expert's tokens
    are padded only up to a multiple of ``block`` and every block multiplies
    against its own expert's weights.  Padded rows are bounded by
    ``num_experts * (block - 1)`` whatever the routing imbalance, and no token
    is dropped.
    """

    def __init__(self, num_experts: int, in_features: int, intermediate: int,
                 beta: float, linear_beta: float, init_std: float, block: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.intermediate = intermediate
        self.block = block
        self.init_std = init_std
        # Same per-expert shapes as nn.Linear(in, 2*inter) and nn.Linear(inter, in).
        self.gate_up = nn.Parameter(torch.empty(num_experts, 2 * intermediate, in_features))
        self.down = nn.Parameter(torch.empty(num_experts, in_features, intermediate))
        # Muon orthogonalizes each expert matrix separately.
        self.gate_up.muon_batched = True
        self.down.muon_batched = True
        self.act = FusedSitu(beta=beta, linear_beta=linear_beta)
        self.reset_structural_init()

    def reset_structural_init(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.gate_up, mean=0.0, std=self.init_std)
            nn.init.normal_(self.down, mean=0.0, std=self.init_std)

    def forward(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        """x: [tokens, D]; topk_idx/topk_weight: [tokens, k]. Returns the weighted mixture [tokens, D]."""
        n_tok, dim = x.shape
        k = topk_idx.shape[1]
        if n_tok == 0:
            return x.new_zeros(0, dim)
        block = self.block
        flat = topk_idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        expert_sorted = flat[order]
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        padded = (counts + block - 1) // block * block
        padded_start = torch.cumsum(padded, 0) - padded
        count_start = torch.cumsum(counts, 0) - counts
        rank_in_expert = torch.arange(flat.numel(), device=x.device) - count_start[expert_sorted]
        dest = padded_start[expert_sorted] + rank_in_expert
        total = int(padded.sum().item())  # one host sync per layer
        blocks = total // block
        block_expert = torch.repeat_interleave(
            torch.arange(self.num_experts, device=x.device), padded // block, output_size=blocks,
        )
        packed = x.new_zeros(total, dim)
        packed = packed.index_copy(0, dest, x[order // k])
        packed = packed.view(blocks, block, dim)
        # Cast the stacked weights once, then pick one matrix per block.
        gate_up = self.gate_up.to(x.dtype).index_select(0, block_expert)
        hidden = torch.bmm(packed, gate_up.transpose(1, 2))
        hidden = self.act(hidden)
        down = self.down.to(x.dtype).index_select(0, block_expert)
        out = torch.bmm(hidden, down.transpose(1, 2)).reshape(total, dim)
        out_sorted = out.index_select(0, dest)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), device=order.device)
        per_slot = out_sorted.index_select(0, inv_order).view(n_tok, k, dim)
        return (per_slot * topk_weight.unsqueeze(-1).to(per_slot.dtype)).sum(dim=1)


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

        # 256 routed experts, stacked.
        self.experts = RoutedExperts(
            self.num_experts,
            in_features=self.latent_dim,
            intermediate=config.moe_intermediate_size,
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
            init_std=config.initializer_range,
            block=getattr(config, "moe_block_size", 64),
        )

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

        # 1. Unconditional shared experts
        shared_out = self.shared_experts[0](x_flat)
        for expert in self.shared_experts[1:]:
            shared_out = shared_out + expert(x_flat)

        # 2. Routed experts run in the latent. Normalize after they are mixed.
        x_lat = self.latent_down(x_flat)

        # 3. Route tokens
        topk_idx, topk_weight = self.gate(x_flat)  # [tokens, k], [tokens, k]

        # 4. Dropless block dispatch; memory does not grow with routing imbalance.
        routed_lat = self.experts(x_lat, topk_idx, topk_weight)

        # 5. RMSNorm sits between the mixture and the up projection.
        routed_out = self.latent_up(self.latent_norm(routed_lat))
        out = routed_out + shared_out
        return out.view(*orig_shape)
