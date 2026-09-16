"""
Loss Spike Protection (SpikeGuard).
Tracks an Exponential Moving Average (EMA) of training loss and skips optimizer updates
if a loss spike or NaN/Inf is detected.
"""

import math
from typing import Tuple, Optional, Dict, Any


class SpikeGuard:
    def __init__(self, spike_factor: float = 1.5, alpha: float = 0.05):
        self.spike_factor = spike_factor
        self.alpha = alpha
        self.ema_loss: Optional[float] = None
        self.total_skips: int = 0
        self.consecutive_skips: int = 0

    def check(self, step: int, loss_val: float) -> Tuple[bool, str]:
        """
        Evaluates current loss against running EMA.
        Returns:
            skip (bool): Whether the optimizer update should be skipped.
            reason (str): Reason for skipping or 'ok'.
        """
        if not math.isfinite(loss_val):
            self.total_skips += 1
            self.consecutive_skips += 1
            return True, f"non-finite loss value: {loss_val}"

        if self.ema_loss is None:
            # Initialize EMA on first valid step
            self.ema_loss = loss_val
            return False, "ok"

        threshold = self.ema_loss * self.spike_factor
        if loss_val > threshold and step > 10:
            self.total_skips += 1
            self.consecutive_skips += 1
            return True, f"loss spike {loss_val:.4f} > {threshold:.4f} (EMA={self.ema_loss:.4f})"

        # Normal healthy step: update EMA
        self.ema_loss = (1.0 - self.alpha) * self.ema_loss + self.alpha * loss_val
        self.consecutive_skips = 0
        return False, "ok"

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_loss": self.ema_loss,
            "total_skips": self.total_skips,
            "consecutive_skips": self.consecutive_skips,
            "spike_factor": self.spike_factor,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ema_loss = state.get("ema_loss")
        self.total_skips = state.get("total_skips", 0)
        self.consecutive_skips = state.get("consecutive_skips", 0)
        self.spike_factor = state.get("spike_factor", 1.5)
        self.alpha = state.get("alpha", 0.05)
