"""
Multi-Head Latent Attention (MLA) Layer.
Compresses KV cache via low-rank projection while preserving full global retrieval.
Divides head dimensions into:
- 128 nope (no positional encoding, reconstructed from latent KV)
- 64 rope (rotary position embedding, shared across heads)
- 128 value (reconstructed from latent KV)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from train.config import MiniK3Config


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 1_048_576, base: float = 10_000_000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, D]
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, L, D]
    sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, L, D]
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: MiniK3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_dim = config.qk_nope_head_dim
        self.qk_rope_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        # Query low-rank compression & expansion
        self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_up = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_nope_dim + self.qk_rope_dim), bias=False)

        # KV low-rank compression & expansion
        self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_up = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_dim + self.v_head_dim), bias=False)
        self.k_rope_proj = nn.Linear(self.hidden_size, self.qk_rope_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        rope_base = getattr(config, "rope_theta", 10_000_000.0)
        self.rotary_emb = RotaryEmbedding(self.qk_rope_dim, config.max_position_embeddings, base=rope_base)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv=None, use_cache: bool = False, cache_position: int = 0,
    ):
        b, l, _ = hidden_states.shape

        # 1. Project Query: [B, L, H * (nope + rope)]
        q_lat = self.q_norm(self.q_down(hidden_states))
        q_full = self.q_up(q_lat).view(b, l, self.num_heads, self.qk_nope_dim + self.qk_rope_dim)
        q_nope = q_full[..., : self.qk_nope_dim]
        q_rope = q_full[..., self.qk_nope_dim :]

        # 2. Project KV: [B, L, H * (nope + value)]
        kv_lat = self.kv_norm(self.kv_down(hidden_states))
        kv_full = self.kv_up(kv_lat).view(b, l, self.num_heads, self.qk_nope_dim + self.v_head_dim)
        k_nope = kv_full[..., : self.qk_nope_dim]
        v = kv_full[..., self.qk_nope_dim :]

        # K RoPE is computed once and shared across heads: [B, L, 1, rope_dim]
        k_rope = self.k_rope_proj(hidden_states).view(b, l, 1, self.qk_rope_dim)
        k_rope = k_rope.expand(-1, -1, self.num_heads, -1)

        # 3. Apply RoPE to rotary portions: [B, H, L, rope_dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        cos, sin = self.rotary_emb(hidden_states, l + cache_position)
        if cache_position:
            cos, sin = cos[cache_position:], sin[cache_position:]
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)

        # 4. Concatenate nope + rope to form full Q and K: [B, H, L, 192]
        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)  # [B, H, L, 128]

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2); v = torch.cat([pv, v], dim=2)
            window = getattr(self.config, "attention_window", 4096)
            k = k[:, :, -window:]; v = v[:, :, -window:]
            # Query attends to all cached keys and current keys; causal mask
            # is only needed within the current token block.
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(weights, v)
            result = self.out_proj(out.transpose(1, 2).contiguous().view(b, l, -1))
            return (result, (k.detach(), v.detach())) if use_cache else result

        # 5. Chunked causal attention. A dense LxL mask cannot fit at 1M;
        # each query attends to a bounded recent KV window.
        window = getattr(self.config, "attention_window", 4096)
        block = min(window, 512)
        outputs = []
        scale = 1.0 / math.sqrt(q.shape[-1])
        for start in range(0, l, block):
            end = min(l, start + block)
            key_start = max(0, end - window)
            qs = q[:, :, start:end]
            ks = k[:, :, key_start:end]
            vs = v[:, :, key_start:end]
            scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
            q_pos = torch.arange(start, end, device=q.device)[:, None]
            k_pos = torch.arange(key_start, end, device=q.device)[None, :]
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append(torch.matmul(weights, vs))
        attn_out = torch.cat(outputs, dim=2)

        # 6. Reshape & Output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, self.num_heads * self.v_head_dim)
        result = self.out_proj(attn_out)
        return (result, (k[:, :, -window:].detach(), v[:, :, -window:].detach())) if use_cache else result
