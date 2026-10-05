"""
Kimi Delta Attention (KDA) Layer.
Linear attention with delta rule recurrence, depthwise short conv (kernel 4),
per-channel forget gates, and Q/K L2-normalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from train.config import MiniK3Config
from train.engine.init_patch import init_dt_bias


class ShortConv1d(nn.Module):
    """
    Depthwise 1D convolution with kernel size 4 and Swish activation.
    Provides local token mixing before the recurrent delta step.
    Supports conv_state caching for bit-exact autoregressive generation.
    """
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, L, D] -> conv expects [B, D, L]
        b, l, d = x.shape
        x_t = x.transpose(1, 2)
        k_minus_1 = self.kernel_size - 1

        if conv_state is not None:
            # Historical tokens are cached in conv_state: [B, D, k-1]
            cat_x = torch.cat([conv_state, x_t], dim=2)
            out = F.conv1d(cat_x, self.conv.weight, self.conv.bias, groups=self.channels, padding=0)
            new_state = cat_x[:, :, -k_minus_1:].detach()
        else:
            out = self.conv(x_t)[:, :, :l]
            if l >= k_minus_1:
                new_state = x_t[:, :, -k_minus_1:].detach()
            else:
                pad = torch.zeros(b, d, k_minus_1 - l, device=x.device, dtype=x.dtype)
                new_state = torch.cat([pad, x_t], dim=2).detach()

        out = F.silu(out)
        return out.transpose(1, 2), new_state


def bounded_log_decay(z: torch.Tensor, a_log: torch.Tensor, g_min: float) -> torch.Tensor:
    """K3 retention logit. Finite z stays strictly inside (g_min, 0)."""
    scale = a_log.exp().view(1, -1, 1, 1)
    return g_min * torch.sigmoid(scale * z)


def chunk_delta_rule(q, k, v, log_decay, beta, state, chunk_size=32):
    """Chunked delta rule (UT transform). Matches the token recurrence

        S_t = S_{t-1} diag(g_t);  S_t += beta_t (v_t - S_t k_t) k_t^T;  o_t = S_t q_t

    with S stored as [value, key] and g_t = exp(log_decay_t) per key channel.

    All intra-chunk work is computed for every chunk at once. Each chunk's
    unit-lower-triangular system is solved with ``solve_triangular`` (no LU,
    no host synchronisation). Only two small matmuls per chunk cross the
    sequential state carry. Peak 5D tensors are [B, H, L/C, C, C, D], i.e.
    O(L * C * D) per head instead of the previous per-chunk loop of LU solves.

    Inputs are float32: q/k/v [B, H, L, D], log_decay [B, H, L, D] (<= 0),
    beta [B, H, L], state [B, H, D, D]. Returns (out [B, H, L, D], state).
    """
    batches, heads, length, dim = q.shape
    vdim = v.shape[-1]
    size = max(1, min(int(chunk_size), length))
    pad = (-length) % size
    if pad:
        # Zero beta + zero log decay at the tail: no write, no decay, so the
        # final state and every real output are unchanged.
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        log_decay = F.pad(log_decay, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
    chunks = (length + pad) // size
    shape = (batches, heads, chunks, size)
    q = q.reshape(*shape, dim)
    k = k.reshape(*shape, dim)
    v = v.reshape(*shape, vdim)
    beta = beta.reshape(*shape)
    cumulative = torch.cumsum(log_decay.reshape(*shape, dim), dim=3)
    cum_g = torch.exp(cumulative)

    tril = torch.ones(size, size, device=q.device, dtype=torch.bool).tril()
    strict = tril.clone().fill_diagonal_(False)
    # Upper (future) log-ratios are positive and can overflow exp(); zero them
    # before exponentiating, then mask the factor itself.
    differences = (cumulative.unsqueeze(4) - cumulative.unsqueeze(3))
    differences = differences.masked_fill(~tril[:, :, None], 0.0)
    factor = torch.exp(differences).masked_fill(~tril[:, :, None], 0.0)
    keyed = factor * k.unsqueeze(3)                            # [B,H,N,C(t),C(j),D]
    dots = torch.einsum("bhntjd,bhntd->bhntj", keyed, k)        # k_t^T G_tj k_j
    key_query = torch.einsum("bhntjd,bhntd->bhntj", keyed, q)   # q_t^T G_tj k_j, j <= t

    eye = torch.eye(size, device=q.device, dtype=q.dtype)
    system = eye + (beta.unsqueeze(-1) * dots).masked_fill(~strict, 0.0)
    rhs = torch.cat([beta.unsqueeze(-1) * v, beta.unsqueeze(-1) * cum_g * k], dim=-1)
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    u_part, w_part = solved.split([vdim, dim], dim=-1)          # written = u - W S^T

    last = cum_g[:, :, :, -1]                                    # [B,H,N,D]
    key_final = k * torch.exp(cumulative[:, :, :, -1:] - cumulative)
    starts = []
    writes = []
    for n in range(chunks):
        starts.append(state)
        written = u_part[:, :, n] - w_part[:, :, n] @ state.transpose(-1, -2)
        writes.append(written)
        state = state * last[:, :, n].unsqueeze(-2) + written.transpose(-1, -2) @ key_final[:, :, n]
    start_states = torch.stack(starts, dim=2)                    # [B,H,N,V,D]
    written_all = torch.stack(writes, dim=2)                     # [B,H,N,C,V]
    from_state = torch.einsum("bhnvd,bhncd->bhncv", start_states, cum_g * q)
    from_writes = torch.einsum("bhntj,bhnjv->bhntv", key_query, written_all)
    out = (from_state + from_writes).reshape(batches, heads, chunks * size, vdim)
    return out[:, :, :length], state


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: MiniK3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_kda_heads
        self.head_dim = config.head_dim
        self.proj_dim = self.num_heads * self.head_dim  # 4 * 128 = 512

        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.out_proj = nn.Linear(self.proj_dim, self.hidden_size, bias=False)
        self.q_proj.weight.muon_heads = self.num_heads
        self.k_proj.weight.muon_heads = self.num_heads
        self.v_proj.weight.muon_heads = self.num_heads
        # Per-head write gate. sigmoid(0) = 0.5 after zero bias init, not a fixed overwrite.
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        # Per-channel output gate, applied in head space before the output projection.
        self.out_gate = nn.Linear(self.hidden_size, self.proj_dim, bias=True)

        # Short depthwise conv for Q, K, V
        self.q_conv = ShortConv1d(self.proj_dim, config.short_conv_kernel_size)
        self.k_conv = ShortConv1d(self.proj_dim, config.short_conv_kernel_size)
        self.v_conv = ShortConv1d(self.proj_dim, config.short_conv_kernel_size)

        # Per-channel forget gate: hidden -> low_rank (num_heads * 16) -> proj_dim
        gate_low_rank = self.num_heads * 16
        self.gate_down = nn.Linear(self.hidden_size, gate_low_rank, bias=False)
        self.gate_up = nn.Linear(gate_low_rank, self.proj_dim, bias=True)
        self.gate_lower_bound = config.gate_lower_bound

        # A_h starts at 0. The per-channel bias is the Kimi Linear dt bias.
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.proj_dim, dtype=torch.float32))
        init_dt_bias(self.dt_bias)
        self.head_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
        conv_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with causal delta rule recurrence.
        hidden_states: [B, L, hidden_size]
        """
        b, l, _ = hidden_states.shape

        q_cs, k_cs, v_cs = conv_state if conv_state is not None else (None, None, None)

        # 1. Linear projections + Short Conv (with conv cache) + Swish
        q, new_q_cs = self.q_conv(self.q_proj(hidden_states), conv_state=q_cs)
        k, new_k_cs = self.k_conv(self.k_proj(hidden_states), conv_state=k_cs)
        v, new_v_cs = self.v_conv(self.v_proj(hidden_states), conv_state=v_cs)
        new_conv_state = (new_q_cs, new_k_cs, new_v_cs)

        # 2. Reshape into heads: [B, H, L, head_dim]
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. L2 Normalization on Q and K (critical for bf16 numerical stability)
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2)  # [B, H, L]
        out_gate = torch.sigmoid(self.out_gate(hidden_states))
        out_gate = out_gate.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Lower-bounded log decay: g_min * sigmoid(exp(A_h) * z), g_min = -5.
        gate_raw = self.gate_up(F.silu(self.gate_down(hidden_states)))  # [B, L, proj_dim]
        gate_raw = gate_raw.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            z = gate_raw.float() + self.dt_bias.float().view(1, self.num_heads, 1, self.head_dim)
            log_decay = bounded_log_decay(z, self.A_log.float(), self.gate_lower_bound)

        # 5. Delta Rule Recurrence
        # State: S of shape [B, H, head_dim, head_dim]
        # Always maintain recurrence state in float32 to prevent FP16 overflow over 2048 steps
        if recurrent_state is None:
            state = torch.zeros(b, self.num_heads, self.head_dim, self.head_dim,
                                device=hidden_states.device, dtype=torch.float32)
        else:
            state = recurrent_state.float()

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            out, state = chunk_delta_rule(
                q.float(), k.float(), v.float(), log_decay, beta.float(), state,
                chunk_size=int(getattr(self.config, "kda_chunk_size", 32)),
            )
            out = self.head_norm(out)
        out = out.to(hidden_states.dtype)
        out = out * out_gate.to(out.dtype)
        out = out.transpose(1, 2).contiguous().view(b, l, self.proj_dim)
        output = self.out_proj(out)
        return output, state, new_conv_state
