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

        # Short depthwise conv for Q, K, V
        self.q_conv = ShortConv1d(self.proj_dim, config.short_conv_kernel_size)
        self.k_conv = ShortConv1d(self.proj_dim, config.short_conv_kernel_size)
        self.v_conv = ShortConv1d(self.proj_dim, config.short_conv_kernel_size)

        # Per-channel forget gate: hidden -> low_rank (num_heads * 16) -> proj_dim
        gate_low_rank = self.num_heads * 16
        self.gate_down = nn.Linear(self.hidden_size, gate_low_rank, bias=False)
        self.gate_up = nn.Linear(gate_low_rank, self.proj_dim, bias=True)
        self.gate_lower_bound = config.gate_lower_bound

        # Recurrence parameters: A_log and dt_bias
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.dt_bias = nn.Parameter(torch.empty(self.proj_dim, dtype=torch.float32))
        init_dt_bias(self.dt_bias)

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

        # 4. Compute forget gate: g in range [lower_bound, 0]
        gate_raw = self.gate_up(F.silu(self.gate_down(hidden_states)))  # [B, L, proj_dim]
        gate_raw = gate_raw.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        # Per-key-channel log decay. A_log must participate in the graph;
        # the old exp(gate) * dt both ignored A_log and nearly erased memory.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            dt = F.softplus(gate_raw.float() + self.dt_bias.float().view(1, self.num_heads, 1, self.head_dim))
            log_decay = -self.A_log.float().exp().view(1, self.num_heads, 1, 1) * dt
            if self.gate_lower_bound is not None:
                log_decay = log_decay.clamp(min=self.gate_lower_bound)
            decay = log_decay.exp()

        # 5. Delta Rule Recurrence
        # State: S of shape [B, H, head_dim, head_dim]
        # Always maintain recurrence state in float32 to prevent FP16 overflow over 2048 steps
        if recurrent_state is None:
            state = torch.zeros(b, self.num_heads, self.head_dim, self.head_dim,
                                device=hidden_states.device, dtype=torch.float32)
        else:
            state = recurrent_state.float()

        outs = []
        # .float() alone does not stop autocast from lowering matmul precision.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            for t in range(l):
                qt, kt, vt = q[:, :, t].float(), k[:, :, t].float(), v[:, :, t].float()
                # State layout is [value, key]; decay acts on the key axis.
                decayed = state * decay[:, :, t].unsqueeze(-2)
                predicted = torch.matmul(decayed, kt.unsqueeze(-1)).squeeze(-1)
                delta = vt - predicted
                state = decayed + delta.unsqueeze(-1) * kt.unsqueeze(-2)
                outs.append(torch.matmul(state, qt.unsqueeze(-1)).squeeze(-1).to(hidden_states.dtype))

        out = torch.stack(outs, dim=2)  # [B, H, L, D]
        out = out.transpose(1, 2).contiguous().view(b, l, self.proj_dim)
        output = self.out_proj(out)
        return output, state, new_conv_state
