"""Switches for the historical 2048 Mini K3 run, CED, and longer contexts.

Defaults preserve the audited pretraining command. Longer sequences and the
dense CED baseline stay off until a flag asks for them.
"""

LONG_CONTEXT_SOFT_LIMIT = 16384


def resolve_run(
    model,
    sequence_length,
    attention_window,
    allow_long_sequence,
    default_sequence,
    default_window,
    max_position,
    peak_lr=None,
    default_peak_lr=6.0e-4,
    init_checkpoint=None,
):
    if model not in ("mini-k3", "ced"):
        raise ValueError(f"Unknown model {model!r}; choose mini-k3 or ced")
    seq = default_sequence if sequence_length is None else int(sequence_length)
    if seq < 1 or seq > int(max_position):
        raise ValueError(f"sequence length {seq} is outside 1..{max_position}")
    if seq > LONG_CONTEXT_SOFT_LIMIT and not allow_long_sequence:
        raise ValueError(
            f"sequence length {seq} is above {LONG_CONTEXT_SOFT_LIMIT}. "
            "1,048,576 is the inference position ceiling. "
            "Continue from the 2048 checkpoint at 4096, 8192, or 16384 first. "
            "Pass --allow-long-sequence only after a shorter run fits in memory."
        )
    if model == "ced":
        window = None
    elif attention_window is None:
        window = int(default_window)
    else:
        window = int(attention_window)
        if window < 1:
            raise ValueError("attention window must be positive")
    if peak_lr is None:
        if init_checkpoint and seq > int(default_sequence):
            raise ValueError(
                "A longer continuation must set --peak-lr. "
                "Use 6e-5, the pretrain minimum, instead of the 6e-4 pretrain peak."
            )
        lr = float(default_peak_lr)
    else:
        lr = float(peak_lr)
    if lr <= 0:
        raise ValueError("peak lr must be positive")
    return {"model": model, "sequence_length": seq, "attention_window": window, "peak_lr": lr}
