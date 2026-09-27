"""One extra prediction head: from h_t and the embedding of token t+1, predict token t+2."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from train.config import MiniK3Config


class MTPBlock(nn.Module):
    def __init__(self, config: MiniK3Config):
        super().__init__()
        hidden = config.hidden_size
        heads = max(1, hidden // config.head_dim)
        self.norm_h = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.norm_e = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.proj = nn.Linear(hidden * 2, hidden, bias=False)
        self.attn_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.mlp = nn.Linear(hidden, hidden * 4, bias=False)
        self.mlp_down = nn.Linear(hidden * 4, hidden, bias=False)
        self.heads = heads
        self.head_dim = hidden // heads

    def step(self, hidden_last, token_embed):
        """Draft the token after `token_embed` from the hidden state that predicted it."""
        x = self.proj(torch.cat([self.norm_h(hidden_last), self.norm_e(token_embed)], dim=-1))
        return self._mix(x)

    def forward(self, hidden, token_embed):
        # hidden/token_embed: [B, L, H]. Position i uses h_i and E(w_{i+1}) to predict w_{i+2}.
        if hidden.size(1) < 3:
            return hidden[:, :0]
        h = self.norm_h(hidden[:, :-2])
        e = self.norm_e(token_embed[:, 1:-1])
        return self._mix(self.proj(torch.cat([h, e], dim=-1)))

    def _mix(self, x):
        residual = x
        qkv = self.qkv(self.attn_norm(x))
        b, length, _ = qkv.shape
        qkv = qkv.view(b, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, length, -1)
        x = residual + self.out(attn)
        x = x + self.mlp_down(F.silu(self.mlp(self.mlp_norm(x))))
        return x
