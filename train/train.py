"""
Main Pretraining Entrypoint for Mini Kimi K3 on 4 Tesla V100 32GB GPUs.

Features:
- Micro-batch gradient accumulation: micro-batch 1 x 2048 tokens x 32 accumulation
  steps x 4 GPUs = 262,144 tokens / optimizer step (38,147 steps = 10,000,007,168 tokens)
- Warmup-Stable-Decay (WSD) scheduler (2% warmup, last 15% linear decay to 10% of peak);
  ``--schedule_total_steps`` lets a short run follow the 10B schedule shape
- Auxiliary-loss-free MoE balancing by per-expert score quantiles (NoAuxBalancer;
  ``balancer_gamma`` is kept only for old call sites and is not used)
- Memory-lean step: DDP ``gradient_as_bucket_view``, LM/MTP loss computed in
  sequence chunks without materialising full logits (``compute_logits=False``)
- Online dead-expert and load imbalance monitoring
- Step time, tokens/sec, MFU and peak-memory telemetry
- EMA Loss Spike Protection (SpikeGuard) and single-check non-finite gradient skip
- Fixed held-out validation slice (same tokens every evaluation), best checkpoint by
  validation backbone LM loss
- Atomic, fsynced checkpoints with bit-exact resume support (including inside decay)
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import inspect
import argparse
import datetime
import dataclasses
import numpy as np
from pathlib import Path
from contextlib import nullcontext
# Permit both ``python -m train.train`` and the documented
# ``python train/train.py`` invocation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataclasses import replace
from train.config import CANONICAL_PARAMETER_COUNTS, DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.run_options import (
    RETIRED_ATTENTION_WINDOW_MESSAGE, resolve_run, resolve_schedule_total_steps,
)
from train.engine.init_patch import assert_initialised
from train.engine.scheduler import wsd_lr, is_in_decay_phase
from train.engine.balancer import NoAuxBalancer
from train.engine.spike_guard import SpikeGuard
from train.engine.checkpoint import CheckpointManager
from train.engine.muon import build_optimizer
from train.data.loader import MultiSourceDataLoader
from train.readiness import check_training_readiness


# Tesla V100-SXM2 (32GB) FP16 Tensor Core theoretical peak per GPU.
V100_PEAK_FLOPS_PER_GPU = 125e12
V100_4X_PEAK_FLOPS = 4 * V100_PEAK_FLOPS_PER_GPU
GIB = float(1 << 30)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Mini Kimi K3 on 4x V100")
    parser.add_argument(
        "--data_manifest",
        type=str,
        default="/data/mini-k3/data/prepared-v2-supplement-v2/manifests/pretrain_stable.json",
        help="Path to the audited pretraining manifest JSON",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="/data/mini-k3/checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--total_steps", type=int, default=38147,
                        help="Optimizer steps this run executes (default 38,147 for 10B tokens)")
    parser.add_argument(
        "--schedule_total_steps", type=int, default=None,
        help=("Length of the WSD learning-rate/data-mix schedule (default: --total_steps). "
              "LR warmup/decay and the decay-mix switch follow this length while the loop "
              "runs --total_steps. Example: --total_steps 200 --schedule_total_steps 38147 "
              "runs the first 200 steps of the real 10B schedule (warmup 762 steps), so the "
              "short run tests the real early LR curve. Must be >= --total_steps."),
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from latest valid checkpoint")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval in steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Telemetry reporting interval in steps")
    parser.add_argument(
        "--validation_manifest",
        type=str,
        default="/data/mini-k3/data/prepared-v2-supplement-v2/manifests/validation.json",
        help="Path to the audited validation manifest JSON",
    )
    parser.add_argument("--validation_interval", type=int, default=500,
                        help="Validate every N steps (also at every checkpoint and the final step)")
    parser.add_argument("--validation_batches", type=int, default=8,
                        help="Micro-batches per rank in the fixed held-out validation slice (default 8)")
    parser.add_argument("--model", choices=("mini-k3", "ced"), default="mini-k3",
                        help="mini-k3 is the pretrained backbone. ced trains the separate dense baseline.")
    parser.add_argument("--sequence-length", type=int, default=None,
                        help="Override config sequence length. Default 2048. 4096, 8192 and 16384 continue long context.")
    # Retired 2026-10-03: the CSA2 cache is exact at any length; passing it is an error.
    parser.add_argument("--attention-window", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-long-sequence", action="store_true",
                        help="Permit a training sequence above 16384. 1048576 remains the inference ceiling.")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Load model weights only and start a new run. Do not combine with --resume.")
    parser.add_argument("--peak-lr", type=float, default=None,
                        help="Override the pretrain peak. Required when continuing at a longer sequence.")
    parser.add_argument("--seed", type=int, default=42, help="Global initialization and data seed (default: 42)")
    parser.add_argument("--compile", action="store_true",
                        help="Opt in to torch.compile; disabled by default on V100/custom attention kernels")
    args = parser.parse_args(argv)
    if args.attention_window is not None:
        parser.error(RETIRED_ATTENTION_WINDOW_MESSAGE)
    if args.validation_batches < 1:
        parser.error("--validation_batches must be positive")
    if args.log_interval < 1 or args.save_interval < 1:
        parser.error("--log_interval and --save_interval must be positive")
    return args


def calculate_mfu(active_params: int, tokens_per_sec: float, peak_flops: float = V100_4X_PEAK_FLOPS) -> float:
    """
    Standard MFU formula: (6 * active_parameters * tokens_per_sec) / peak_flops
    """
    return (6.0 * active_params * tokens_per_sec) / peak_flops


def config_digest(cfg) -> str:
    """Stable hash of every model/training config field (run-signature binding)."""
    try:
        payload = dataclasses.asdict(cfg)
    except TypeError:
        payload = dict(vars(cfg))
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_run_signature(opts, cfg, args, world_size, readiness, param_counts, schedule_total_steps):
    """Everything that must match for a resume to continue the same experiment."""
    return {
        "model": opts["model"],
        "total_steps": args.total_steps,
        "schedule_total_steps": schedule_total_steps,
        "sequence_length": cfg.sequence_length,
        "peak_lr": cfg.peak_lr,
        "world_size": world_size,
        "micro_batch_size": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "seed": args.seed,
        "warmup_frac": getattr(cfg, "warmup_frac", None),
        "decay_frac": getattr(cfg, "decay_frac", None),
        "min_lr_frac": getattr(cfg, "min_lr_frac", None),
        "weight_decay": getattr(cfg, "weight_decay", None),
        "grad_clip": getattr(cfg, "grad_clip", None),
        "precision": getattr(cfg, "precision", None),
        "stable_mix": dict(getattr(cfg, "stable_mix", {}) or {}),
        "decay_mix": dict(getattr(cfg, "decay_mix", {}) or {}),
        "muon_update_scale": getattr(cfg, "muon_update_scale", None),
        "moe_block_size": getattr(cfg, "moe_block_size", None),
        "loss_chunk_size": getattr(cfg, "loss_chunk_size", None),
        "config_sha256": config_digest(cfg),
        "train_manifest_sha256": readiness["manifest"]["sha256"],
        "validation_manifest_sha256": readiness["validation"]["sha256"],
        "parameter_counts": param_counts,
    }


def phase_mix(cfg, phase: str):
    return cfg.decay_mix if phase == "decay" else cfg.stable_mix


def _supports_chunked_loss(model) -> bool:
    target = model.module if isinstance(model, DDP) else model
    target = getattr(target, "_orig_mod", target)
    try:
        return "compute_logits" in inspect.signature(target.forward).parameters
    except (TypeError, ValueError):
        return False


def _call_model(model, x, y):
    """Training/eval forward without full logits (chunked LM + MTP loss).

    Mini K3 takes ``compute_logits=False``; the dense CED baseline has no such
    switch and no separate LM loss, so its total loss is its LM loss.
    """
    if _supports_chunked_loss(model):
        out = model(x, labels=y, compute_logits=False)
    else:
        out = model(x, labels=y)
    if out.get("lm_loss") is None:
        out = dict(out)
        out["lm_loss"] = out["loss"]
    return out


@torch.no_grad()
def run_validation(raw_model, val_loader, snapshot, batches, device, amp_dtype, use_amp,
                   distributed, world_size):
    """Score the fixed held-out slice; returns (lm_loss, mtp_loss or None).

    The loader is rewound to ``snapshot`` first, so every evaluation of a run
    scores exactly the same tokens and successive values are comparable.
    """
    was_training = raw_model.training
    raw_model.eval()
    val_loader.load_state_dict(snapshot)
    sums = torch.zeros(3, device=device, dtype=torch.float32)  # lm sum, mtp sum, mtp count
    try:
        for _ in range(batches):
            vx, vy = val_loader.next_batch(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = _call_model(raw_model, vx, vy)
            sums[0] += out["lm_loss"].detach().float()
            if out.get("mtp_loss") is not None:
                sums[1] += out["mtp_loss"].detach().float()
                sums[2] += 1.0
    finally:
        raw_model.train(was_training)
        val_loader.load_state_dict(snapshot)
    if distributed:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    lm, mtp_sum, mtp_count = sums.tolist()
    lm = lm / (batches * world_size)
    mtp = (mtp_sum / mtp_count) if mtp_count > 0 else None
    return lm, mtp


def _fmt(value, spec=".4f"):
    return "n/a" if value is None else format(value, spec)


def main(argv=None):
    args = parse_args(argv)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume continues one run. --init-checkpoint starts a new run from weights. Pass only one.")
    cfg = replace(DEFAULT_CONFIG)
    opts = resolve_run(
        args.model, args.sequence_length, args.allow_long_sequence,
        cfg.sequence_length, cfg.max_position_embeddings,
        args.peak_lr, cfg.peak_lr, args.init_checkpoint,
    )
    cfg.sequence_length = opts["sequence_length"]
    cfg.peak_lr = opts["peak_lr"]
    cfg.validate()
    schedule_total_steps = resolve_schedule_total_steps(args.total_steps, args.schedule_total_steps)
    world_size_for_contract = int(os.environ.get("WORLD_SIZE", "1"))
    expected_tokens = (
        args.total_steps * cfg.micro_batch_size * cfg.gradient_accumulation_steps
        * world_size_for_contract * cfg.sequence_length
    )
    readiness = check_training_readiness(
        args.data_manifest,
        args.validation_manifest,
        vocab_size=cfg.vocab_size,
        stable_mix=cfg.stable_mix,
        expected_parameters=CANONICAL_PARAMETER_COUNTS.get(opts["model"]),
        expected_training_tokens=expected_tokens,
    )
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # All ranks construct identical weights. Only Python/NumPy loader streams
    # are rank-offset so source choices differ across DDP workers.
    torch.manual_seed(args.seed)
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if distributed:
        if not torch.cuda.is_available(): raise RuntimeError("4-card training requires CUDA")
        torch.cuda.set_device(local_rank)
        # Checkpoint writes of a 1.15B model + optimizer and the first-step
        # kernel warm-up can exceed NCCL's default 10-minute watchdog.
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    world_size = dist.get_world_size() if distributed else 1
    seq_len = getattr(cfg, "sequence_length", cfg.max_position_embeddings)
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and cfg.precision == "fp16"),
        init_scale=1024.0,
        growth_interval=2000,
    )

    if rank == 0:
        print("=" * 80)
        print(f" Mini Kimi K3 training  model={opts['model']}  sequence={cfg.sequence_length}")
        print(f" Target Hardware: 4x Tesla V100-SXM2 (32GB), FP16")
        print("=" * 80)

    if device.type != "cuda" and rank == 0:
        print("[!] WARNING: CUDA is not available. Training will be extremely slow on CPU!")

    # 1. Initialize Model
    if rank == 0: print(f"[*] Building model {opts['model']}...")
    if opts["model"] == "mini-k3":
        model = MiniK3ForCausalLM(cfg).to(device)
    else:
        from train.models.deepseek_coder import DeepSeekCoderConfig, DeepSeekCoderForCausalLM
        ced_cfg = DeepSeekCoderConfig(vocab_size=cfg.vocab_size, max_seq_len=cfg.sequence_length)
        model = DeepSeekCoderForCausalLM(ced_cfg).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        init_file = init_path / "model.pt" if init_path.is_dir() else init_path
        if not init_file.is_file():
            raise FileNotFoundError(f"Init checkpoint not found: {init_file}")
        if rank == 0:
            print(f"[*] Loading initial weights from {init_file}")
        model.load_state_dict(torch.load(init_file, map_location="cpu", weights_only=False))

    # Optional PyTorch 2.0 compile on supported CUDA
    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        print("[*] Compiling model with torch.compile()...")
        # compile with default mode (avoids breaking custom autograd)
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[!] torch.compile skipped: {e}")

    # 2. Parameter Accounting
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if distributed:
        # find_unused_parameters=True stays on for mini-k3: the vision tower
        # gets no gradient on text-only batches. gradient_as_bucket_view makes
        # .grad alias the all-reduce buckets instead of a second full copy.
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
                    find_unused_parameters=(opts["model"] == "mini-k3"),
                    gradient_as_bucket_view=True)
    param_counts = raw_model.count_parameters()
    total_params = param_counts["total"]
    active_params = param_counts["active"]
    if rank == 0: print(f"[*] World size: {world_size}; Parameters: Total = {total_params / 1e6:.2f}M | Active = {active_params / 1e6:.2f}M | Non-Embed Active = {param_counts['non_embed_active'] / 1e6:.2f}M")

    # Matrix weights use per-head Muon. Embeddings, gains, and biases use AdamW.
    # The router correction bias is not trainable and is left out.
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    opt_kwargs = {}
    if "muon_update_scale" in inspect.signature(build_optimizer).parameters and hasattr(cfg, "muon_update_scale"):
        opt_kwargs["muon_update_scale"] = cfg.muon_update_scale
    optimizer = build_optimizer(
        iter(trainable),
        lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        **opt_kwargs,
    )

    # 4. Engine Utilities
    balancer = NoAuxBalancer(raw_model, gamma=cfg.balancer_gamma)
    spike_guard = SpikeGuard(spike_factor=1.5, alpha=0.05)
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=3,
        save_interval=args.save_interval,
        milestone_interval=5000,
    )
    data_loader = MultiSourceDataLoader(
        manifest_path=args.data_manifest,
        seq_len=seq_len,
        batch_size=cfg.micro_batch_size,
        allow_missing=False,
        rank=rank,
        world_size=world_size,
    )
    if not data_loader.streams:
        raise RuntimeError(f"Training manifest has no readable shards: {args.data_manifest}")
    val_loader = MultiSourceDataLoader(
        manifest_path=args.validation_manifest, seq_len=seq_len, batch_size=cfg.micro_batch_size,
        rank=rank, world_size=world_size,
    ) if Path(args.validation_manifest).is_file() else None
    # Fixed held-out slice: every evaluation rewinds to this cursor.
    val_snapshot = val_loader.state_dict() if val_loader is not None else None
    data_loader.set_mix(cfg.stable_mix)
    phase = "stable"
    run_signature = build_run_signature(opts, cfg, args, world_size, readiness, param_counts,
                                        schedule_total_steps)

    # 5. Resume or Start Fresh
    start_step = 0
    total_tokens_seen = 0
    if args.resume:
        meta = ckpt_manager.load_latest(raw_model, optimizer, spike_guard, rank=rank,
                                        world_size=world_size,
                                        scaler=scaler, expected_run_signature=run_signature)
        if meta:
            saved_step = int(meta.get("step", 0))
            start_step = saved_step + 1
            total_tokens_seen = meta.get("total_tokens_seen", 0)
            # The checkpoint was written after step ``saved_step`` ran, so its
            # loader state belongs to that step's phase. Select that mix before
            # loading (set_mix keeps counters when the mix is unchanged) and
            # let the loop switch at the real boundary.
            phase = "decay" if is_in_decay_phase(saved_step, schedule_total_steps, cfg.decay_frac) else "stable"
            data_loader.set_mix(phase_mix(cfg, phase))
            saved_loader = meta.get("data_loader")
            if not saved_loader:
                raise RuntimeError("Checkpoint has no data loader state for this rank; refusing inexact resume")
            data_loader.load_state_dict(saved_loader.get("train", saved_loader))
            if not data_loader.same_mix(phase_mix(cfg, phase)):
                raise RuntimeError(
                    f"Resumed loader mix {data_loader.target_mix()} does not match the {phase} mix at step {saved_step}"
                )
            if rank == 0:
                print(f"[*] Resumed successfully at Step {start_step} (Tokens seen: {total_tokens_seen:,}; "
                      f"phase={phase}; best val lm={_fmt(ckpt_manager.best_metric)} @ {ckpt_manager.best_step})")
        else:
            if rank == 0: print("[*] No checkpoint found. Starting from Step 0.")

    # 6. Step 0 Sanity Check (if starting from 0)
    if start_step == 0 and not args.init_checkpoint:
        if rank == 0: print("[*] Performing Step 0 initialization check...")
        assert_initialised(raw_model)
        probe_cursor = data_loader.state_dict()
        probe_random_state = random.getstate()
        was_training = raw_model.training
        raw_model.eval()
        with torch.no_grad():
            x0, y0 = data_loader.next_batch(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out0 = _call_model(raw_model, x0, y0)
            probe = torch.zeros(3, device=device, dtype=torch.float32)
            probe[0] = out0["lm_loss"].detach().float()
            if out0.get("mtp_loss") is not None:
                probe[1] = out0["mtp_loss"].detach().float()
                probe[2] = 1.0
        del out0, x0, y0
        raw_model.train(was_training)
        data_loader.load_state_dict(probe_cursor)
        random.setstate(probe_random_state)
        if distributed:
            dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        lm0_sum, mtp0_sum, mtp0_count = probe.tolist()
        loss0 = lm0_sum / world_size
        mtp0 = mtp0_sum / mtp0_count if mtp0_count > 0 else None
        expected_loss = math.log(cfg.vocab_size)
        ok = 11.90 <= loss0 <= 12.25
        if rank == 0:
            print(f"[*] Step 0 LM loss: {loss0:.4f} | MTP: {_fmt(mtp0)} "
                  f"(Theoretical uniform ln(vocab) = {expected_loss:.4f})")
        if not ok:
            raise ValueError(f"Initial LM loss {loss0:.4f} outside expected range [11.90, 12.25]. Check initialization, labels, and vocab.")
        if rank == 0:
            print("    -> PASS: Initial loss conforms strictly to uniform random initialization.")
        if distributed:
            dist.barrier()

    # 7. Main Training Loop
    tokens_per_step = cfg.micro_batch_size * cfg.gradient_accumulation_steps * world_size * seq_len
    if rank == 0:
        print("\n" + "-" * 80)
        print(f" Starting pretraining from step {start_step} to {args.total_steps} "
              f"(schedule length {schedule_total_steps}, warmup {max(1, int(schedule_total_steps * cfg.warmup_frac))}, "
              f"decay starts {int(schedule_total_steps * (1.0 - cfg.decay_frac))})")
        print(f" Micro-batch: {cfg.micro_batch_size} ({cfg.micro_batch_size * seq_len:,} tokens)")
        print(f" Gradient accumulation: {cfg.gradient_accumulation_steps} steps")
        print(f" Effective step batch: {tokens_per_step:,} tokens")
        print(f" Validation: {args.validation_batches} batches/rank x {world_size} ranks (fixed slice), "
              f"every {args.validation_interval} steps, at checkpoints and at the final step")
        print("-" * 80 + "\n")

    model.train()
    peak_flops = world_size * V100_PEAK_FLOPS_PER_GPU
    interval_time = 0.0
    interval_steps = 0
    GA = cfg.gradient_accumulation_steps
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step, args.total_steps):
        step_start_time = time.time()

        # Update learning rate (WSD schedule, possibly longer than this run)
        lr = wsd_lr(step, schedule_total_steps, cfg.peak_lr, cfg.warmup_frac, cfg.decay_frac, cfg.min_lr_frac)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Check for phase transition into decay
        if is_in_decay_phase(step, schedule_total_steps, cfg.decay_frac) and phase != "decay":
            phase = "decay"
            data_loader.set_mix(cfg.decay_mix)
            if rank == 0: print(f"\n>>> [Step {step}] ENTERING DECAY/ANNEAL PHASE: Switching data mix to math/code emphasis. <<<\n")

        # Accumulate gradients across micro-batches
        optimizer.zero_grad(set_to_none=True)
        # [total loss, lm loss, mtp loss, mtp count, aux loss, aux count] summed over micro-steps
        sums = torch.zeros(6, device=device, dtype=torch.float32)

        for micro_step in range(GA):
            is_last_micro_step = (micro_step == GA - 1)
            sync_context = nullcontext() if (not distributed or is_last_micro_step) else model.no_sync()
            with sync_context:
                x, y = data_loader.next_batch(device)
                # V100 uses FP16; GradScaler prevents underflow/overflow.
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = _call_model(model, x, y)
                loss = outputs["loss"]
                scaler.scale(loss / GA).backward()
            sums[0] += loss.detach().float()
            sums[1] += outputs["lm_loss"].detach().float()
            if outputs.get("mtp_loss") is not None:
                sums[2] += outputs["mtp_loss"].detach().float()
                sums[3] += 1.0
            if outputs.get("aux_loss") is not None:
                sums[4] += outputs["aux_loss"].detach().float()
                sums[5] += 1.0
            del outputs, loss, x, y

        if distributed:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        loss_sum, lm_sum, mtp_sum, mtp_count, aux_sum, aux_count = sums.tolist()
        step_loss = loss_sum / (GA * world_size)
        step_lm = lm_sum / (GA * world_size)
        step_mtp = mtp_sum / mtp_count if mtp_count > 0 else None
        step_aux = aux_sum / aux_count if aux_count > 0 else None

        # Spike detection uses the backbone LM loss: MTP and the CSA indexer KL
        # are auxiliary and would only add noise to the EMA test.
        skip_update, spike_reason = spike_guard.check(step, step_lm)
        # One global finiteness check: the pre-clip total norm is non-finite iff
        # some gradient element is. Gradients are identical on every rank after
        # the DDP all-reduce; the MAX all-reduce only guards against divergence.
        scaler.unscale_(optimizer)
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip)
        nonfinite_t = (~torch.isfinite(grad_norm_t)).to(torch.int32).reshape(1)
        if distributed:
            dist.all_reduce(nonfinite_t, op=dist.ReduceOp.MAX)
        nonfinite = bool(nonfinite_t.item())
        grad_norm = float(grad_norm_t)
        if nonfinite:
            if skip_update:
                spike_reason = f"{spike_reason}; non-finite gradient"
            else:
                skip_update, spike_reason = spike_guard.record_skip(
                    f"non-finite gradient (norm={grad_norm}, scale={scaler.get_scale():.0f})", step)
        if skip_update:
            if rank == 0:
                print(f"[!] Step {step} SKIPPED: {spike_reason}")
            abort = spike_guard.abort_reason()
            if abort:
                raise RuntimeError(f"Aborting: {abort}; last reason: {spike_reason}")
            optimizer.zero_grad(set_to_none=True)
        else:
            scaler.step(optimizer)
            spike_guard.record_ok(step)
        # Always update: a non-finite step lowers the loss scale.
        scaler.update()

        # Update load balancer
        telemetry = balancer.step()

        # Synchronize GPU for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        step_duration = time.time() - step_start_time
        interval_time += step_duration
        interval_steps += 1
        total_tokens_seen += tokens_per_step

        is_final = step == args.total_steps - 1
        early = step - start_step < 2
        is_log_step = step % args.log_interval == 0 or is_final or early
        if is_log_step:
            # Peak memory since the last log, max over ranks. Every rank takes
            # this branch (step-based), so the collective is safe.
            mem = torch.zeros(2, device=device, dtype=torch.float64)
            if device.type == "cuda":
                mem[0] = torch.cuda.max_memory_allocated(device)
                mem[1] = torch.cuda.max_memory_reserved(device)
                if distributed:
                    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
                torch.cuda.reset_peak_memory_stats(device)
            avg_step = interval_time / max(1, interval_steps)
            tokens_per_sec = tokens_per_step / max(1e-9, avg_step)
            mfu = calculate_mfu(active_params, tokens_per_sec, peak_flops)
            if rank == 0:
                print(
                    f"[Step {step:05d}/{args.total_steps}] "
                    f"loss {step_loss:.4f} | lm {step_lm:.4f} | mtp {_fmt(step_mtp)} | "
                    + (f"aux {step_aux:.4f} | " if step_aux is not None else "")
                    + f"lr {lr:.2e} | gnorm(pre-clip) {grad_norm:.3f} | "
                    f"step {avg_step:.2f}s | tok/s {tokens_per_sec:,.0f} | MFU {mfu * 100:.1f}% | "
                    f"mem max alloc {mem[0].item() / GIB:.2f} GiB / reserved {mem[1].item() / GIB:.2f} GiB | "
                    f"dead {telemetry['dead_frac'] * 100:.1f}% | imb {telemetry['imbalance']:.2f} | "
                    f"scale {scaler.get_scale():.0f} | skips {spike_guard.total_skips}"
                )
                mix = data_loader.realised_mix()
                mix_text = " ".join(f"{name}={share:.3f}" for name, share in mix.items())
                print(f"    mix: {mix_text}", flush=True)
            interval_time = 0.0
            interval_steps = 0

        is_save_step = (step > 0 and step % args.save_interval == 0) or is_final
        val_lm = val_mtp = None
        if val_loader is not None and (
            (args.validation_interval > 0 and step % args.validation_interval == 0) or is_save_step
        ):
            val_lm, val_mtp = run_validation(
                raw_model, val_loader, val_snapshot, args.validation_batches, device, amp_dtype, use_amp,
                distributed, world_size,
            )
            if rank == 0:
                print(f"[*] Validation at step {step}: lm {val_lm:.4f} | mtp {_fmt(val_mtp)} "
                      f"({args.validation_batches * world_size} fixed sequences)", flush=True)

        # Periodic Checkpoint saving
        if is_save_step:
            meta_info = {
                "step": step,
                "total_tokens_seen": total_tokens_seen,
                "loss": step_loss,
                "lm_loss": step_lm,
                "mtp_loss": step_mtp,
                "lr": lr,
                "phase": phase,
                "schedule_total_steps": schedule_total_steps,
            }
            if val_lm is not None:
                # Best-checkpoint selection uses the backbone next-token loss.
                meta_info["val_loss"] = val_lm
                meta_info["val_lm_loss"] = val_lm
                meta_info["val_mtp_loss"] = val_mtp
            saved_path = ckpt_manager.save(
                step=step,
                model=raw_model,
                optimizer=optimizer,
                spike_guard=spike_guard,
                data_loader_state={
                    "train": data_loader.state_dict(),
                    "validation": val_snapshot,
                },
                extra_meta={**meta_info, "run_signature": run_signature},
                rank=rank,
                world_size=world_size,
                scaler=scaler,
            )
            if rank == 0: print(f"[*] Checkpoint saved at step {step}: {saved_path}", flush=True)

    if distributed: dist.barrier(); dist.destroy_process_group()
    if rank == 0:
        print("\n" + "=" * 80); print(f" Training Complete! Total tokens: {total_tokens_seen:,}"); print("=" * 80)


if __name__ == "__main__":
    main()
