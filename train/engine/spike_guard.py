"""
Loss Spike Protection (SpikeGuard).
Tracks an Exponential Moving Average (EMA) of training loss and skips optimizer updates
if a loss spike or NaN/Inf is detected.

Protocol for one optimizer step (``train.py``)::

    skip, reason = guard.check(step, loss)          # loss-based decision
    ... unscale + grad norm ...
    if grad_norm is non-finite and not skip:
        skip, reason = guard.record_skip("non-finite gradient", step)
    if not skip:
        guard.record_ok(step)                       # update applied

A step is counted at most once in ``total_skips``/``consecutive_skips``.
``consecutive_skips`` is only reset by ``record_ok`` -- i.e. once an update
has really been applied -- so a run of steps with finite loss but overflowing
gradients still reaches the abort threshold.

Abort policy (``abort_reason``): FP16 gradient overflow is routine while the
GradScaler searches for its scale (each overflow halves it), so overflow skips
abort only after ``max_consecutive_skips`` (20, i.e. the scale fell 2^20).
Loss-spike skips (finite loss far above the EMA) abort after
``max_consecutive_spikes`` (5). Both counters reset when an update is applied.
"""

import math
from typing import Tuple, Optional, Dict, Any


class SpikeGuard:
    def __init__(self, spike_factor: float = 1.5, alpha: float = 0.05, warmup_steps: int = 10,
                 max_consecutive_skips: int = 20, max_consecutive_spikes: int = 5):
        self.spike_factor = spike_factor
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.max_consecutive_skips = max_consecutive_skips
        self.max_consecutive_spikes = max_consecutive_spikes
        self.ema_loss: Optional[float] = None
        self.total_skips: int = 0
        self.consecutive_skips: int = 0
        self.consecutive_spikes: int = 0
        self._skipped_step: Optional[int] = None
        self._pending: Optional[Tuple[int, float]] = None

    def _skip(self, step: Optional[int], reason: str, spike: bool = False) -> Tuple[bool, str]:
        if step is None or step != self._skipped_step:
            self.total_skips += 1
            self.consecutive_skips += 1
            if spike:
                self.consecutive_spikes += 1
        self._skipped_step = step
        self._pending = None
        return True, reason

    def check(self, step: int, loss_val: float) -> Tuple[bool, str]:
        """
        Evaluates current loss against running EMA.
        Returns:
            skip (bool): Whether the optimizer update should be skipped.
            reason (str): Reason for skipping or 'ok'.
        The EMA is updated only when the update is committed by ``record_ok``.
        """
        if not math.isfinite(loss_val):
            return self._skip(step, f"non-finite loss value: {loss_val}")

        if self.ema_loss is not None:
            threshold = self.ema_loss * self.spike_factor
            if loss_val > threshold and step > self.warmup_steps:
                return self._skip(
                    step, f"loss spike {loss_val:.4f} > {threshold:.4f} (EMA={self.ema_loss:.4f})",
                    spike=True,
                )
        self._pending = (step, float(loss_val))
        return False, "ok"

    def record_skip(self, reason: str, step: Optional[int] = None) -> Tuple[bool, str]:
        """Record a bad gradient discovered after the loss check.

        If ``check`` already skipped this ``step`` the skip is not counted again.
        """
        return self._skip(step, reason)

    def record_ok(self, step: Optional[int] = None) -> None:
        """The optimizer update of ``step`` was applied: fold its loss into the EMA."""
        pending = self._pending
        self._pending = None
        if pending is not None and (step is None or pending[0] == step):
            loss_val = pending[1]
            if self.ema_loss is None:
                self.ema_loss = loss_val
            else:
                self.ema_loss = (1.0 - self.alpha) * self.ema_loss + self.alpha * loss_val
        self.consecutive_skips = 0
        self.consecutive_spikes = 0

    def abort_reason(self) -> Optional[str]:
        """Non-None when training must stop (see module docstring)."""
        if self.consecutive_spikes >= self.max_consecutive_spikes:
            return f"{self.consecutive_spikes} consecutive loss-spike skips"
        if self.consecutive_skips >= self.max_consecutive_skips:
            return f"{self.consecutive_skips} consecutive skipped updates (FP16 overflow or non-finite loss)"
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_loss": self.ema_loss,
            "total_skips": self.total_skips,
            "consecutive_skips": self.consecutive_skips,
            "consecutive_spikes": self.consecutive_spikes,
            "max_consecutive_skips": self.max_consecutive_skips,
            "max_consecutive_spikes": self.max_consecutive_spikes,
            "spike_factor": self.spike_factor,
            "alpha": self.alpha,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ema_loss = state.get("ema_loss")
        self.total_skips = state.get("total_skips", 0)
        self.consecutive_skips = state.get("consecutive_skips", 0)
        self.consecutive_spikes = state.get("consecutive_spikes", 0)
        self.max_consecutive_skips = state.get("max_consecutive_skips", self.max_consecutive_skips)
        self.max_consecutive_spikes = state.get("max_consecutive_spikes", self.max_consecutive_spikes)
        self.spike_factor = state.get("spike_factor", 1.5)
        self.alpha = state.get("alpha", 0.05)
        self.warmup_steps = state.get("warmup_steps", 10)
        self._skipped_step = None
        self._pending = None
