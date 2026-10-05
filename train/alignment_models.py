"""Checkpoint resolution plus reward/value scalar-head models for alignment.

Checkpoint layouts that are accepted (see ``train/engine/checkpoint.py``):

* a pretraining run root containing ``step_XXXXXX/`` directories: the newest
  directory that has a ``COMPLETE`` marker is used, otherwise loading fails;
* a single ``step_XXXXXX/`` directory (must carry ``COMPLETE`` when it has a
  ``meta.pt``) or an alignment output directory with ``alignment_meta.json``;
* a bare ``model.pt`` file (its directory is still searched for metadata).

The trained sequence length / attention window are read from the pretraining
``meta.pt`` ``run_signature`` or from ``alignment_meta.json``. They are never
guessed silently: callers get ``None`` when no metadata exists.

Scalar heads (``RewardModel`` / ``ValueModel``) are loaded explicitly:

* ``source="causal"`` initialises the backbone from a causal LM ``model.pt``
  and leaves the score head freshly initialised;
* ``source="scalar"`` requires that model's own artifact
  (``reward_model.pt`` or ``value_model.pt``). A value model can never pick up
  reward weights by accident and vice versa.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from train.models.mini_k3 import MiniK3ForCausalLM


MODEL_FILE = "model.pt"
REWARD_FILE = "reward_model.pt"
VALUE_FILE = "value_model.pt"
ALIGNMENT_META = "alignment_meta.json"


@dataclass
class CheckpointInfo:
    model_file: Optional[Path]
    directory: Path
    meta: Optional[dict]
    alignment_meta: Optional[dict]

    @property
    def run_signature(self) -> dict:
        if self.meta and isinstance(self.meta.get("run_signature"), dict):
            return self.meta["run_signature"]
        return {}

    @property
    def sequence_length(self) -> Optional[int]:
        value = self.run_signature.get("sequence_length")
        if value is None and self.alignment_meta:
            value = self.alignment_meta.get("sequence_length")
        return int(value) if value is not None else None

    @property
    def attention_window(self) -> Optional[int]:
        value = self.run_signature.get("attention_window")
        if value is None and self.alignment_meta:
            value = self.alignment_meta.get("attention_window")
        return int(value) if value is not None else None


def _step_number(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_complete_step(root: str | Path) -> Path:
    """Newest ``step_*`` directory carrying ``COMPLETE``; fail when there is none."""
    root = Path(root)
    steps = [d for d in root.glob("step_*") if d.is_dir() and _step_number(d) >= 0]
    complete = sorted((d for d in steps if (d / "COMPLETE").is_file()), key=_step_number)
    if not complete:
        partial = ", ".join(sorted(d.name for d in steps)) or "none"
        raise FileNotFoundError(
            f"{root} has no complete step_* checkpoint (COMPLETE marker missing; found: {partial})"
        )
    return complete[-1]


def resolve_checkpoint(path: str | Path, artifact: str = MODEL_FILE,
                       require_artifact: bool = True) -> CheckpointInfo:
    p = Path(path)
    if p.is_file():
        model_file, directory = p, p.parent
    elif p.is_dir():
        directory = p
        known = (MODEL_FILE, REWARD_FILE, VALUE_FILE, "meta.pt", ALIGNMENT_META)
        if not any((p / name).exists() for name in known):
            directory = latest_complete_step(p)
        model_file = directory / artifact
        if not model_file.is_file():
            if require_artifact:
                raise FileNotFoundError(f"{directory} has no {artifact}")
            model_file = None
    else:
        raise FileNotFoundError(f"checkpoint not found: {p}")
    if (directory / "meta.pt").is_file() and directory.name.startswith("step_") \
            and not (directory / "COMPLETE").is_file():
        raise FileNotFoundError(f"{directory} is a partial checkpoint (no COMPLETE marker)")
    meta = None
    if (directory / "meta.pt").is_file():
        meta = torch.load(directory / "meta.pt", map_location="cpu", weights_only=False)
    alignment_meta = None
    if (directory / ALIGNMENT_META).is_file():
        alignment_meta = json.loads((directory / ALIGNMENT_META).read_text(encoding="utf-8"))
    return CheckpointInfo(model_file, directory, meta, alignment_meta)


def config_for_checkpoint(info: CheckpointInfo, base=None):
    """Copy ``base`` (default ``DEFAULT_CONFIG``) with the checkpoint's trained geometry."""
    if base is None:
        from train.config import DEFAULT_CONFIG
        base = DEFAULT_CONFIG
    model_name = info.run_signature.get("model")
    if model_name not in (None, "mini-k3"):
        raise ValueError(f"checkpoint was trained as {model_name!r}; only mini-k3 weights load here")
    cfg = replace(base)
    if info.sequence_length is not None:
        cfg.sequence_length = info.sequence_length
    if info.attention_window is not None:
        cfg.attention_window = info.attention_window
    return cfg


def _load_state(model_file: Path) -> dict:
    return torch.load(model_file, map_location="cpu", weights_only=False)


def _is_scalar_state(state: dict) -> bool:
    return any(str(key).startswith("backbone.") for key in state)


def load_causal_model(checkpoint: str | Path, device=None, config=None):
    """Return ``(model, info, config)`` for a causal LM checkpoint (strict load)."""
    info = resolve_checkpoint(checkpoint, MODEL_FILE)
    cfg = config if config is not None else config_for_checkpoint(info)
    model = MiniK3ForCausalLM(cfg)
    state = _load_state(info.model_file)
    if _is_scalar_state(state):
        raise ValueError(f"{info.model_file} is a reward/value checkpoint, not a causal LM")
    model.load_state_dict(state, strict=True)
    if device is not None:
        model = model.to(device)
    return model, info, cfg


class ScalarHeadModel(nn.Module):
    """Backbone plus a scalar score (per sequence, or per token for values)."""

    ARTIFACT = None

    def __init__(self, config, checkpoint=None, source: str = "causal"):
        super().__init__()
        self.backbone = MiniK3ForCausalLM(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        # A 1 x hidden head is not a matrix Muon should orthogonalise.
        self.score.weight.adam_only = True
        if checkpoint:
            self.load_checkpoint(checkpoint, source)

    def load_checkpoint(self, checkpoint, source: str) -> Path:
        if source == "causal":
            info = resolve_checkpoint(checkpoint, MODEL_FILE)
            state = _load_state(info.model_file)
            if _is_scalar_state(state):
                raise ValueError(
                    f"{info.model_file} is a scalar-head checkpoint; pass source='scalar' explicitly"
                )
            self.backbone.load_state_dict(state, strict=True)
            return info.model_file
        if source != "scalar":
            raise ValueError("source must be 'causal' or 'scalar'")
        if self.ARTIFACT is None:
            raise ValueError("scalar checkpoints need a RewardModel or ValueModel")
        p = Path(checkpoint)
        if p.is_dir():
            model_file = p / self.ARTIFACT
            if not model_file.is_file():
                raise FileNotFoundError(
                    f"{p} has no {self.ARTIFACT}; {type(self).__name__} will not load other artifacts"
                )
        else:
            model_file = p
            other = {REWARD_FILE, VALUE_FILE} - {self.ARTIFACT}
            if p.name in other:
                raise ValueError(f"{p} is not a {type(self).__name__} artifact ({self.ARTIFACT})")
        state = _load_state(model_file)
        if not _is_scalar_state(state) or "score.weight" not in state:
            raise ValueError(f"{model_file} does not contain backbone.* and score.weight")
        own = {str(key).removeprefix("backbone."): value
               for key, value in state.items() if str(key).startswith("backbone.")}
        self.backbone.load_state_dict(own, strict=True)
        self.score.load_state_dict({"weight": state["score.weight"]}, strict=True)
        return model_file

    def forward(self, input_ids, attention_mask=None, per_token: bool = False):
        # MiniK3 layers consume the four-stream/mHC state. Calling a layer
        # with one tensor bypasses the actual backbone contract, so use the
        # complete model and pool its final hidden states.
        hidden = self.backbone(
            input_ids, return_hidden=True, compute_logits=False,
        )["hidden"]
        if per_token:
            return self.score(hidden).squeeze(-1).float()
        if attention_mask is None:
            idx = torch.full((hidden.size(0),), hidden.size(1)-1, device=hidden.device, dtype=torch.long)
        else:
            if attention_mask.shape != input_ids.shape or not attention_mask.any(-1).all():
                raise ValueError("attention_mask must match input_ids and retain a token in every row")
            positions = torch.arange(hidden.size(1), device=hidden.device)
            idx = positions.expand_as(input_ids).masked_fill(~attention_mask.bool(), -1).max(-1).values
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.score(pooled).squeeze(-1).float()


class RewardModel(ScalarHeadModel):
    ARTIFACT = REWARD_FILE


class ValueModel(ScalarHeadModel):
    ARTIFACT = VALUE_FILE
