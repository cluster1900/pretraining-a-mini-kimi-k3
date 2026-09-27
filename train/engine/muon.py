"""Per-head Muon for matrix weights, AdamW for embeddings, biases, and gains.

Attention projections tagged with `muon_heads` are orthogonalized one head at a
time. Embedding tables tagged `adam_only` stay on AdamW. The router bias is
not trainable and never enters this optimizer.
"""

import torch
from torch.optim import Optimizer


def newton_schulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    updated = matrix.float()
    updated = updated / (updated.norm() + 1e-7)
    transposed = updated.size(0) > updated.size(1)
    if transposed:
        updated = updated.T
    for _ in range(steps):
        gram = updated @ updated.T
        updated = a * updated + (b * gram + c * gram @ gram) @ updated
    if transposed:
        updated = updated.T
    return updated


def orthogonalize(grad: torch.Tensor, heads: int, steps: int) -> torch.Tensor:
    if heads and grad.ndim == 2 and grad.shape[0] % heads == 0:
        grouped = grad.reshape(heads, grad.shape[0] // heads, grad.shape[1])
        parts = [newton_schulz(grouped[i], steps) for i in range(heads)]
        return torch.stack(parts, dim=0).reshape_as(grad)
    return newton_schulz(grad, steps)


class HybridOptimizer(Optimizer):
    def __init__(self, params, lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8,
                 muon_momentum=0.95, ns_steps=5):
        unique = []
        seen = set()
        for param in params:
            if param is None or not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
                        muon_momentum=muon_momentum, ns_steps=ns_steps)
        super().__init__(unique, defaults)

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
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                use_muon = param.ndim == 2 and not getattr(param, "adam_only", False)
                if use_muon:
                    buf = state.get("momentum")
                    if buf is None:
                        buf = torch.zeros_like(grad)
                        state["momentum"] = buf
                    buf.mul_(momentum).add_(grad)
                    update = orthogonalize(buf, int(getattr(param, "muon_heads", 0) or 0), steps)
                    if decay:
                        param.mul_(1 - lr * decay)
                    param.add_(update.to(param.dtype), alpha=-lr)
                    continue
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step_buf = state.get("step")
                step_id = 1 if step_buf is None else int(step_buf.item()) + 1
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
                state["step"] = torch.tensor(step_id, dtype=torch.long)
        return loss


def build_optimizer(params, lr: float, weight_decay: float = 0.1) -> HybridOptimizer:
    return HybridOptimizer(params, lr=lr, weight_decay=weight_decay)
