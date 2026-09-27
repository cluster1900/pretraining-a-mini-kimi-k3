"""E2M1 FP4 KV cache. V100 has no FP4 tensor cores, so this packs values in software.

Training activations stay FP16. Only the inference latent is quantized.
"""

import torch

# Absolute magnitudes of the E2M1 grid. The sign is stored in the high bit.
FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class PackedKV:
    """Two FP4 values per byte, plus one FP16 scale per token."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor):
        self.packed = packed
        self.scale = scale

    @property
    def shape(self):
        batch, tokens, half = self.packed.shape
        return torch.Size((batch, tokens, half * 2))

    def clone(self):
        return PackedKV(self.packed.detach().clone(), self.scale.detach().clone())

    def detach(self):
        return PackedKV(self.packed.detach(), self.scale.detach())

    def to(self, device):
        return PackedKV(self.packed.to(device), self.scale.to(device))


def quantize_fp4(values: torch.Tensor) -> PackedKV:
    """Quantize the last dimension. It must be even. Scale is the token max."""
    if values.shape[-1] % 2 != 0:
        raise ValueError("FP4 packing needs an even last dimension")
    magnitudes = values.new_tensor(FP4_MAGNITUDES)
    scale = values.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / magnitudes[-1]
    unit = (values.detach() / scale).clamp(-magnitudes[-1], magnitudes[-1])
    index = (unit.abs().unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    code = index + (unit < 0).to(index.dtype) * 8
    paired = code.view(*code.shape[:-1], -1, 2)
    packed = (paired[..., 0] + (paired[..., 1] << 4)).to(torch.uint8)
    return PackedKV(packed, scale.to(torch.float16))


def dequantize_fp4(blob: PackedKV) -> torch.Tensor:
    magnitudes = blob.scale.new_tensor(FP4_MAGNITUDES)
    low = (blob.packed & 0x0F).long()
    high = (blob.packed >> 4).long()
    code = torch.stack((low, high), dim=-1).reshape(*blob.packed.shape[:-1], -1)
    sign = torch.where(code >= 8, -1.0, 1.0).to(blob.scale.dtype)
    magnitude = magnitudes[code % 8]
    return sign * magnitude * blob.scale.to(magnitude.dtype)
