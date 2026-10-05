"""Switches for the historical 2048 Mini K3 run, CED, and longer contexts.

Defaults preserve the audited pretraining command. Longer sequences and the
dense CED baseline stay off until a flag asks for them.

The MLA sliding ``attention_window`` switch was retired (2026-10-03): the CSA2
cache is exact at any length, so no window is part of a run any more.
"""

LONG_CONTEXT_SOFT_LIMIT = 16384

RETIRED_ATTENTION_WINDOW_MESSAGE = (
    "--attention-window was retired: the model no longer uses a sliding MLA window "
    "(the CSA2 cache is exact at any length). Remove the flag."
)


def resolve_run(
    model,
    sequence_length,
    allow_long_sequence,
    default_sequence,
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
    if seq > int(default_sequence) and not init_checkpoint:
        raise ValueError(
            "A sequence longer than the 2048 pretraining length must start from "
            "an explicit --init-checkpoint. Train 2048 first, then continue in a "
            "separate checkpoint directory."
        )
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
    return {"model": model, "sequence_length": seq, "peak_lr": lr}


def resolve_schedule_total_steps(total_steps, schedule_total_steps=None):
    """LR/decay schedule length; defaults to the number of steps this run executes.

    A short run may borrow the shape of a longer schedule (e.g. 200 steps of
    the 38,147-step 10B schedule run the real 762-step warmup prefix). The
    schedule may not be shorter than the run: steps past its end would sit at
    the minimum LR forever.
    """
    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError("--total_steps must be positive")
    if schedule_total_steps is None:
        return total_steps
    schedule = int(schedule_total_steps)
    if schedule < total_steps:
        raise ValueError(
            f"--schedule_total_steps={schedule} is shorter than --total_steps={total_steps}"
        )
    return schedule
