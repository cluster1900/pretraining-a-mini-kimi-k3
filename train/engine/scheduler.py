"""
Warmup-Stable-Decay (WSD) Learning Rate Scheduler.
Provides constant learning rate during the long stable phase, and linear decay in the final 15%.
"""

from typing import Tuple


def wsd_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_frac: float = 0.02,
    decay_frac: float = 0.15,
    min_lr_frac: float = 0.10,
) -> float:
    """
    Computes learning rate for given step in WSD schedule:
    - Warmup: linear increase from 0 to peak_lr over total_steps * warmup_frac
    - Stable: constant peak_lr
    - Decay: linear decrease from peak_lr to peak_lr * min_lr_frac over last decay_frac
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))
    decay_start_step = int(total_steps * (1.0 - decay_frac))
    
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    elif step < decay_start_step:
        return peak_lr
    else:
        decay_steps = max(1, total_steps - decay_start_step)
        progress = (step - decay_start_step) / decay_steps
        progress = min(1.0, max(0.0, progress))
        return peak_lr * (1.0 - (1.0 - min_lr_frac) * progress)


def is_in_decay_phase(step: int, total_steps: int, decay_frac: float = 0.15) -> bool:
    """Returns True if the training has entered the final decay/anneal phase."""
    decay_start_step = int(total_steps * (1.0 - decay_frac))
    return step >= decay_start_step
