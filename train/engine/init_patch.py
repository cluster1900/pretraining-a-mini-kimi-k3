"""
Initialization patches for Mini K3.
Fixes uninitialized memory in KDA dt_bias and provides finite value pre-flight checks.
"""

import math
import torch
import torch.nn as nn


def init_dt_bias(param: nn.Parameter, dt_min: float = 1e-3, dt_max: float = 1e-1, floor: float = 1e-4) -> None:
    """
    Initialize timestep bias (dt_bias) in Mamba / GatedDeltaNet style.
    Uniform sampling in log space, then inverted through softplus.
    """
    with torch.no_grad():
        dt = torch.exp(
            torch.rand(param.shape, device=param.device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=floor)
        # Inverse softplus: softplus_inv(dt) = dt + log(-expm1(-dt))
        val = dt + torch.log(-torch.expm1(-dt))
        param.copy_(val.to(param.dtype))


def assert_initialised(model: nn.Module) -> None:
    """
    Pre-flight check: asserts that all floating point parameters and buffers
    are strictly finite (no inf or NaN) before Step 0 begins.
    """
    bad = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_floating_point() and not torch.isfinite(t).all():
            bad.append(name)
    if bad:
        raise ValueError(f"Non-finite values detected in model parameters/buffers before training: {bad[:10]}")


def apply_init_patches(model: nn.Module) -> None:
    """
    Scans the model and applies proper initialization to:
    1. KDA dt_bias parameters (previously uninitialized torch.empty)
    2. Router e_score_correction_bias (ensure initialized to zeros and requires_grad=False)
    """
    # PyTorch Embedding defaults to N(0, 1); with a tied LM head this makes
    # self-token logits hundreds of times too large for causal pretraining.
    std = getattr(getattr(model, "config", None), "initializer_range", 0.02)
    initialized = set()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if id(module.weight) not in initialized:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    initialized.add(id(module.weight))
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Conv1d) and module.bias is not None:
                nn.init.zeros_(module.bias)
    for name, module in model.named_modules():
        # Patch KDA dt_bias
        if hasattr(module, "dt_bias") and isinstance(module.dt_bias, nn.Parameter):
            init_dt_bias(module.dt_bias)
        
        # Patch MoE Gate bias
        if hasattr(module, "e_score_correction_bias") and isinstance(module.e_score_correction_bias, torch.Tensor):
            with torch.no_grad():
                module.e_score_correction_bias.zero_()
            module.e_score_correction_bias.requires_grad_(False)
    
    # Run pre-flight check
    assert_initialised(model)
