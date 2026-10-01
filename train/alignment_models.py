from pathlib import Path
import torch
import torch.nn as nn
from train.models.mini_k3 import MiniK3ForCausalLM


class ScalarHeadModel(nn.Module):
    """Backbone plus a scalar score per sequence (reward or value model)."""
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.backbone = MiniK3ForCausalLM(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        if checkpoint:
            ckpt_p = Path(checkpoint)
            if ckpt_p.is_dir():
                reward_file = ckpt_p / "reward_model.pt"
                model_file = reward_file if reward_file.is_file() else ckpt_p / "model.pt"
            else:
                model_file = ckpt_p
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            # A causal checkpoint contains bare backbone keys; an RM checkpoint
            # contains ``backbone.*`` plus the scalar head.
            if any(str(key).startswith("backbone.") for key in state):
                own = {str(key).removeprefix("backbone."): value
                       for key, value in state.items() if str(key).startswith("backbone.")}
                self.backbone.load_state_dict(own, strict=True)
                if "score.weight" in state:
                    self.score.load_state_dict({"weight": state["score.weight"]}, strict=True)
            else:
                self.backbone.load_state_dict(state, strict=True)

    def forward(self, input_ids, attention_mask=None):
        # MiniK3 layers consume the four-stream/mHC state. Calling a layer
        # with one tensor bypasses the actual backbone contract, so use the
        # complete model and pool its final hidden states.
        hidden = self.backbone(
            input_ids, return_hidden=True, compute_logits=False,
        )["hidden"]
        if attention_mask is None:
            idx = torch.full((hidden.size(0),), hidden.size(1)-1, device=hidden.device, dtype=torch.long)
        else:
            if attention_mask.shape != input_ids.shape or not attention_mask.any(-1).all():
                raise ValueError("attention_mask must match input_ids and retain a token in every row")
            positions = torch.arange(hidden.size(1), device=hidden.device)
            idx = positions.expand_as(input_ids).masked_fill(~attention_mask.bool(), -1).max(-1).values
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.score(pooled).squeeze(-1)


class RewardModel(ScalarHeadModel):
    pass


class ValueModel(ScalarHeadModel):
    pass
