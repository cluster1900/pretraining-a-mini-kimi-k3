"""
situ activation with a recompute-in-backward autograd function.

situ(gate, up) = beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)

The FP32 intermediates (tanh, sigmoid, clamped up) are not saved for backward;
only the [N, 2F] input is kept and the activation is recomputed in backward.
Output magnitude is bounded by beta * linear_beta (4 * 25 = 100), safe in FP16.

History: an earlier revision carried an unused Triton kernel and a "saves
~40GB" claim. The kernel was never called and was removed (2026-10-03); the
saving is only the FP32 intermediates of this activation.
"""

import torch
import torch.nn as nn
from typing import Optional


def _situ_forward_pytorch(x: torch.Tensor, beta: float = 4.0, linear_beta: Optional[float] = 25.0) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)

    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


class SituRecomputeFunction(torch.autograd.Function):
    """Saves only the input; recomputes the FP32 intermediates in backward."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float, linear_beta: Optional[float]):
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        ctx.save_for_backward(x)
        with torch.no_grad():
            return _situ_forward_pytorch(x, beta, linear_beta)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            xd = x.detach().requires_grad_(True)
            y = _situ_forward_pytorch(xd, ctx.beta, ctx.linear_beta)
            (gx,) = torch.autograd.grad(y, xd, grad_out)
        return gx, None, None


class FusedSitu(nn.Module):
    def __init__(self, beta: float = 4.0, linear_beta: float = 25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SituRecomputeFunction.apply(x, self.beta, self.linear_beta)
