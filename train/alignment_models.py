from pathlib import Path
import torch
import torch.nn as nn
from train.models.mini_k3 import MiniK3ForCausalLM


class ScalarHeadModel(nn.Module):
    """Backbone plus a scalar score per sequence (reward or value model)."""
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.backbone = MiniK3ForCausalLM(config)
        if checkpoint:
            ckpt_p = Path(checkpoint)
            model_file = ckpt_p / "model.pt" if ckpt_p.is_dir() else ckpt_p
            self.backbone.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=False), strict=True)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.backbone.embed_tokens(input_ids)
        for layer in self.backbone.layers:
            hidden, *_ = layer(hidden)
        hidden = self.backbone.norm(hidden)
        if attention_mask is None:
            idx = torch.full((hidden.size(0),), hidden.size(1)-1, device=hidden.device, dtype=torch.long)
        else:
            idx = attention_mask.long().sum(-1).clamp_min(1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.score(pooled).squeeze(-1)


class RewardModel(ScalarHeadModel):
    pass


class ValueModel(ScalarHeadModel):
    pass
