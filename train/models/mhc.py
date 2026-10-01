"""Single-pass manifold-constrained hyper-connections.

Four residual streams stay live across the stack. A sublayer consumes the
pre-mix predicted by the previous sublayer, and from its own input it writes
the doubly-stochastic residual mix, the post-mix, and the next pre-mix.
"""

import torch
import torch.nn as nn


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Project a square logit matrix onto the doubly-stochastic manifold."""
    values = logits.float().softmax(dim=-1)
    for _ in range(iters):
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = values / values.sum(dim=-2, keepdim=True).clamp_min(1e-8)
    return values


def collapse_streams(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return (streams * pre.unsqueeze(-1)).sum(dim=2)


class MHCCoeffs(nn.Module):
    def __init__(self, hidden_size: int, streams: int, sinkhorn_iters: int):
        super().__init__()
        self.streams = streams
        self.sinkhorn_iters = sinkhorn_iters
        self.res = nn.Linear(hidden_size, streams * streams)
        self.post = nn.Linear(hidden_size, streams)
        self.next_pre = nn.Linear(hidden_size, streams)
        self.reset_structural_init()

    def reset_structural_init(self) -> None:
        """Identity residual, write to stream 0, next read from stream 0."""
        with torch.no_grad():
            nn.init.zeros_(self.res.weight)
            nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.next_pre.weight)
            eye = torch.eye(self.streams, dtype=self.res.bias.dtype, device=self.res.bias.device)
            self.res.bias.copy_(eye.reshape(-1) * 8)
            self.post.bias.zero_()
            self.post.bias[0] = 8
            self.next_pre.bias.zero_()
            self.next_pre.bias[0] = 8

    def mix(self, streams: torch.Tensor, update: torch.Tensor, coeff_input: torch.Tensor,
            previous_pre: torch.Tensor | None = None):
        batch, length, width, _ = streams.shape
        res = sinkhorn(self.res(coeff_input).view(batch, length, width, width), self.sinkhorn_iters)
        res = res.to(streams.dtype)
        post = torch.softmax(self.post(coeff_input).float(), dim=-1).to(streams.dtype)
        nxt = torch.softmax(self.next_pre(coeff_input).float(), dim=-1)
        if previous_pre is not None:
            # Keep the learned initial/readout distribution in the graph.  A
            # previous implementation overwrote it before the first collapse,
            # leaving ``mhc_pre`` permanently unused.
            nxt = 0.5 * (nxt + previous_pre.float())
        nxt = nxt.to(streams.dtype)
        mixed = torch.einsum("bsij,bsjd->bsid", res, streams)
        mixed = mixed + post.unsqueeze(-1) * update.unsqueeze(2)
        return mixed, nxt
