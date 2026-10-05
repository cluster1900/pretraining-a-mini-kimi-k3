"""Per-head Muon for matrix weights, AdamW for embeddings, biases, and gains.

Attention projections tagged with `muon_heads` are orthogonalized one head at a
time. Stacked expert weights tagged `muon_batched` ([E, m, n]) are
orthogonalized one expert at a time. Embedding tables tagged `adam_only` stay
on AdamW. The router bias is not trainable and never enters this optimizer.

Update scale: Newton-Schulz returns a matrix whose per-element RMS is about
1/sqrt(max(m, n)). Following Moonlight (Kimi's Muon), every orthogonalized
matrix is multiplied by ``update_scale * sqrt(max(m, n))`` with
``update_scale = 0.2`` so its RMS matches a typical AdamW step and both
optimizers can share the same learning rate and weight decay.
"""

import math
import torch
from torch.optim import Optimizer


def newton_schulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize the last two dims; leading dims are independent matrices."""
    a, b, c = 3.4445, -4.7750, 2.0315
    updated = matrix.float()
    norm = updated.norm(dim=(-2, -1), keepdim=True)
    updated = updated / (norm + 1e-7)
    transposed = updated.size(-2) > updated.size(-1)
    if transposed:
        updated = updated.transpose(-2, -1)
    for _ in range(steps):
        gram = updated @ updated.transpose(-2, -1)
        updated = a * updated + (b * gram + c * gram @ gram) @ updated
    if transposed:
        updated = updated.transpose(-2, -1)
    return updated


def orthogonalize(grad: torch.Tensor, heads: int, steps: int, batched: bool = False,
                  update_scale: float = 0.0) -> torch.Tensor:
    """Return the Muon direction. ``update_scale=0`` keeps the raw NS output."""
    if batched and grad.ndim == 3:
        blocks = grad
    elif heads and grad.ndim == 2 and grad.shape[0] % heads == 0:
        blocks = grad.reshape(heads, grad.shape[0] // heads, grad.shape[1])
    else:
        blocks = grad
    update = newton_schulz(blocks, steps)
    if update_scale:
        update = update * (update_scale * math.sqrt(max(blocks.shape[-2], blocks.shape[-1])))
    return update.reshape_as(grad)


class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8,
                 muon_momentum=0.95, ns_steps=5, muon_update_scale=0.2):
        unique = []
        seen = set()
        for param in params:
            if param is None or not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
                        muon_momentum=muon_momentum, ns_steps=ns_steps,
                        muon_update_scale=muon_update_scale)
        super().__init__(unique, defaults)

    @staticmethod
    def uses_muon(param: torch.Tensor) -> bool:
        if getattr(param, "adam_only", False):
            return False
        return param.ndim == 2 or (param.ndim == 3 and getattr(param, "muon_batched", False))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            momentum = group["muon_momentum"]
            steps = group["ns_steps"]
            scale = group.get("muon_update_scale", 0.0)
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if self.uses_muon(param):
                    buf = state.get("momentum")
                    if buf is None:
                        buf = torch.zeros_like(grad)
                        state["momentum"] = buf
                    buf.mul_(momentum).add_(grad)
                    update = orthogonalize(
                        buf, int(getattr(param, "muon_heads", 0) or 0), steps,
                        batched=bool(getattr(param, "muon_batched", False)),
                        update_scale=scale,
                    )
                    if decay:
                        param.mul_(1 - lr * decay)
                    param.add_(update.to(param.dtype), alpha=-lr)
                    continue
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step_id = int(state.get("step_count", 0)) + 1
                if exp_avg is None:
                    exp_avg = torch.zeros_like(grad)
                    exp_avg_sq = torch.zeros_like(grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                corrected = exp_avg / (1 - beta1 ** step_id)
                second = exp_avg_sq / (1 - beta2 ** step_id)
                param.addcdiv_(corrected, second.sqrt().add_(eps), value=-lr)
                if decay and param.ndim >= 2:
                    param.mul_(1 - lr * decay)
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                # A Python int avoids one device->host sync per AdamW tensor per step.
                state["step_count"] = step_id
        return loss


def build_optimizer(params, lr: float, weight_decay: float = 0.1,
                    muon_update_scale: float = 0.2) -> HybridOptimizer:
    return HybridOptimizer(params, lr=lr, weight_decay=weight_decay,
                           muon_update_scale=muon_update_scale)

