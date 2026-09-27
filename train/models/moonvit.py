"""Scaled MoonViT-V2.

The full K3 tower is 27 layers and about 401M parameters. This one keeps the
same pieces at the text model's scale: RMSNorm, bias-free projections, patch
embedding, spatial attention, temporal attention, a 2x2 pixel shuffle, and a
projector into the language hidden size. Text batches do not call it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ViTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.RMSNorm(hidden)
        self.up = nn.Linear(hidden, hidden * 4, bias=False)
        self.down = nn.Linear(hidden * 4, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        mix = F.scaled_dot_product_attention(q, k, v)
        mix = mix.transpose(1, 2).reshape(batch, length, hidden)
        x = x + self.out(mix)
        x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return x


class MoonViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.vision_hidden
        self.patch = config.vision_patch
        self.hidden = hidden
        self.embed = nn.Conv2d(3, hidden, kernel_size=self.patch, stride=self.patch, bias=False)
        self.spatial = nn.ModuleList(
            _ViTBlock(hidden, config.vision_heads) for _ in range(config.vision_layers)
        )
        self.temporal = _ViTBlock(hidden, config.vision_heads)
        self.shuffle = nn.Linear(hidden * 4, hidden, bias=False)
        self.projector = nn.Linear(hidden, config.hidden_size, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError("images must be [B, 3, H, W] or [B, T, 3, H, W]")
        batch, frames, _, height, width = images.shape
        if height % (self.patch * 2) or width % (self.patch * 2):
            raise ValueError("image sides must be divisible by twice the patch size")
        flat = images.reshape(batch * frames, 3, height, width)
        tokens = self.embed(flat)
        grid_h, grid_w = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        for block in self.spatial:
            tokens = block(tokens)
        tokens = tokens.view(batch, frames, grid_h * grid_w, self.hidden).permute(0, 2, 1, 3)
        tokens = tokens.reshape(batch * grid_h * grid_w, frames, self.hidden)
        tokens = self.temporal(tokens)
        tokens = tokens.view(batch, grid_h, grid_w, frames, self.hidden).permute(0, 3, 1, 2, 4)
        cells = tokens.view(batch, frames, grid_h // 2, 2, grid_w // 2, 2, self.hidden)
        cells = cells.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, frames * (grid_h // 2) * (grid_w // 2), self.hidden * 4)
        return self.projector(self.shuffle(cells))
