"""Full Attention Residuals.

At 12 layers the block-compressed form is unnecessary. Each layer reads a
softmax over the embedding and every previous layer output. A zero query
weights those sources uniformly, so the initial residual stream stays near
the embedding.
"""

import torch
import torch.nn as nn


class AttentionResidual(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.key_norm = nn.RMSNorm(hidden_size, eps=eps)

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        values = torch.stack(sources, dim=2)
        keys = self.key_norm(values)
        logits = torch.einsum("bsnd,d->bsn", keys, self.query)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bsn,bsnd->bsd", weights, values)
