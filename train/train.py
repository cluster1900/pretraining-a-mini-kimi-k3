"""
Main Pretraining Entrypoint for Mini Kimi K3 on 4 Tesla V100 32GB GPUs.

Features:
- Micro-batch gradient accumulation (16,384 tokens * 8 = 131,072 tokens / step)
- Warmup-Stable-Decay (WSD) scheduler
- No-Auxiliary-Loss MoE balancing (gamma=1e-2)
- Online dead-expert and load imbalance monitoring
- Real-time MFU and tokens/sec telemetry
- EMA Loss Spike Protection (SpikeGuard)
- Atomic checkpoints with bit-exact resume support
"""

import os
import sys
import time
import math
import argparse
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

from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.engine.init_patch import assert_initialised
from train.engine.scheduler import wsd_lr, is_in_decay_phase
from train.engine.balancer import NoAuxBalancer
from train.engine.spike_guard import SpikeGuard
from train.engine.checkpoint import CheckpointManager
from train.data.loader import MultiSourceDataLoader


# 4x Tesla V100-SXM2 (32GB) FP16 Tensor Core theoretical peak: 4 * 125 TFLOP/s = 500 TFLOP/s
V100_4X_PEAK_FLOPS = 500e12
V100_4X_HOURLY_COST = 2.40  # Estimated ~$2.40 / hour for 4x V100 instance


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mini Kimi K3 on 4x V100")
    parser.add_argument("--data_manifest", type=str, default="/data/mini-k3/data/manifest.json", help="Path to data manifest JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/mini-k3/checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--total_steps", type=int, default=38147, help="Total training steps (default 38,147 for 5B tokens)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest valid checkpoint")
    parser.add_argument("--allow_missing_data", action="store_true", help="Allow missing data shards (default False)")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval in steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Telemetry reporting interval in steps")
    parser.add_argument("--validation_manifest", type=str, default="/data/mini-k3/data/manifests/validation.json")
    parser.add_argument("--validation_interval", type=int, default=500)
    return parser.parse_args()


def calculate_mfu(active_params: int, tokens_per_sec: float, peak_flops: float = V100_4X_PEAK_FLOPS) -> float:
    """
    Standard MFU formula: (6 * active_parameters * tokens_per_sec) / peak_flops
    """
    return (6.0 * active_params * tokens_per_sec) / peak_flops


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.validate()
    if not args.data_manifest or not Path(args.data_manifest).is_file():
        raise FileNotFoundError(f"Training manifest is required and must exist: {args.data_manifest}")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available(): raise RuntimeError("4-card training requires CUDA")
        torch.cuda.set_device(local_rank); dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seq_len = getattr(cfg, "sequence_length", cfg.max_position_embeddings)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.precision == "fp16"))

    print("=" * 80)
    print(f" Mini Kimi K3 (1.02B) Pretraining Engine")
    if rank == 0: print(f" Target Hardware: 4x Tesla V100-SXM2 (32GB), FP16")
    print("=" * 80)

    if device.type != "cuda" and rank == 0:
        print("[!] WARNING: CUDA is not available. Training will be extremely slow on CPU!")

    # 1. Initialize Model
    if rank == 0: print(f"[*] Building model {cfg.model_name}...")
    model = MiniK3ForCausalLM(cfg).to(device)
    
    # Optional PyTorch 2.0 compile on supported CUDA
    if device.type == "cuda" and hasattr(torch, "compile"):
        print("[*] Compiling model with torch.compile()...")
        # compile with default mode (avoids breaking custom autograd)
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[!] torch.compile skipped: {e}")

    # 2. Parameter Accounting
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if distributed:
        # find_unused_parameters=True is mandatory for MoE because routed experts with 0 tokens have no grad
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    param_counts = raw_model.count_parameters()
    total_params = param_counts["total"]
    active_params = param_counts["active"]
    if rank == 0: print(f"[*] World size: {dist.get_world_size() if distributed else 1}; Parameters: Total = {total_params / 1e6:.2f}M | Active = {active_params / 1e6:.2f}M | Non-Embed Active = {param_counts['non_embed_active'] / 1e6:.2f}M")

    # 3. Setup Optimizer (Excluding MoE correction biases)
    decay_params = []
    no_decay_params = []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        # Biases and layernorms are not weight-decayed
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
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
    world_size = dist.get_world_size() if distributed else 1
    data_loader = MultiSourceDataLoader(
        manifest_path=args.data_manifest,
        seq_len=seq_len,
        batch_size=cfg.micro_batch_size,
        allow_missing=args.allow_missing_data,
        rank=rank,
        world_size=world_size,
    )
    val_loader = MultiSourceDataLoader(manifest_path=args.validation_manifest, seq_len=seq_len,
                                       batch_size=cfg.micro_batch_size) if Path(args.validation_manifest).is_file() else None
    random.seed(random.getstate()[1][0] + rank)
    if data_loader.streams:
        data_loader.set_mix(cfg.stable_mix)

    # 5. Resume or Start Fresh
    start_step = 0
    total_tokens_seen = 0
    if args.resume:
        meta = ckpt_manager.load_latest(raw_model, optimizer, spike_guard, rank=rank,
                                        world_size=(dist.get_world_size() if distributed else 1))
        if meta:
            start_step = meta.get("step", 0) + 1
            total_tokens_seen = meta.get("total_tokens_seen", 0)
            if "data_loader" in meta and meta["data_loader"]:
                data_loader.load_state_dict(meta["data_loader"])
            if rank == 0: print(f"[*] Resumed successfully at Step {start_step} (Tokens seen: {total_tokens_seen:,})")
        else:
            if rank == 0: print("[*] No checkpoint found. Starting from Step 0.")

    # 6. Step 0 Sanity Check (if starting from 0)
    if start_step == 0:
        print("[*] Performing Step 0 initialization check...")
        assert_initialised(raw_model)
        model.eval()
        with torch.no_grad():
            x0, y0 = data_loader.next_batch(device)
            out0 = model(x0, labels=y0)
            loss0 = out0["loss"].item()
            expected_loss = math.log(cfg.vocab_size)
            if rank == 0: print(f"[*] Step 0 Loss: {loss0:.4f} (Theoretical uniform ln(vocab) = {expected_loss:.4f})")
            if not (11.90 <= loss0 <= 12.25):
                raise ValueError(f"Initial loss {loss0:.4f} outside expected range [11.90, 12.25]. Check initialization, labels, and vocab.")
            elif rank == 0:
                print("    -> PASS: Initial loss conforms strictly to uniform random initialization.")
        if distributed: dist.barrier()

    # 7. Main Training Loop
    if rank == 0: print("\n" + "-" * 80)
    print(f" Starting pretraining from step {start_step} to {args.total_steps}...")
    if rank == 0:
        print(f" Micro-batch: {cfg.micro_batch_size} ({cfg.micro_batch_size * seq_len:,} tokens)")
        print(f" Gradient accumulation: {cfg.gradient_accumulation_steps} steps")
        print(f" Effective step batch: {cfg.micro_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if distributed else 1) * seq_len:,} tokens")
    print("-" * 80 + "\n")

    model.train()
    tokens_per_step = cfg.micro_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if distributed else 1) * seq_len
    start_wall_time = time.time()
    last_log_time = time.time()
    phase = "stable"

    for step in range(start_step, args.total_steps):
        step_start_time = time.time()

        # Update learning rate (WSD schedule)
        lr = wsd_lr(step, args.total_steps, cfg.peak_lr, cfg.warmup_frac, cfg.decay_frac, cfg.min_lr_frac)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Check for phase transition into decay
        if is_in_decay_phase(step, args.total_steps, cfg.decay_frac) and phase != "decay":
            phase = "decay"
            if data_loader.streams:
                data_loader.set_mix(cfg.decay_mix)
            if rank == 0: print(f"\n>>> [Step {step}] ENTERING DECAY/ANNEAL PHASE: Switching data mix to math/code emphasis. <<<\n")

        # Accumulate gradients across micro-batches
        optimizer.zero_grad()
        step_loss = 0.0

        for micro_step in range(cfg.gradient_accumulation_steps):
            is_last_micro_step = (micro_step == cfg.gradient_accumulation_steps - 1)
            sync_context = nullcontext() if (not distributed or is_last_micro_step) else model.no_sync()
            with sync_context:
                x, y = data_loader.next_batch(device)
                # V100 uses FP16; GradScaler prevents underflow/overflow.
                amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(x, labels=y)
                    loss = outputs["loss"] / cfg.gradient_accumulation_steps

                scaler.scale(loss).backward()
            step_loss += loss.item()

        # Check for loss spike or NaN
        skip_update, spike_reason = spike_guard.check(step, step_loss)
        if distributed:
            # Every rank must make the same optimizer decision, otherwise
            # DDP replicas silently diverge after a local spike.
            skip_flag = torch.tensor([int(skip_update)], device=device, dtype=torch.int32)
            dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
            if bool(skip_flag.item()) and not skip_update:
                skip_update, spike_reason = True, "peer rank reported non-finite/spike loss"

        if skip_update:
            if rank == 0: print(f"[!] Step {step} SKIPPED: {spike_reason}")
            optimizer.zero_grad()
            grad_norm = 0.0
        else:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=cfg.grad_clip).item()
            scaler.step(optimizer)
            scaler.update()

        # Update load balancer
        telemetry = balancer.step()

        # Synchronize GPU for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        step_duration = time.time() - step_start_time
        tokens_per_sec = tokens_per_step / max(1e-5, step_duration)
        mfu = calculate_mfu(active_params, tokens_per_sec)
        total_tokens_seen += tokens_per_step

        # Telemetry logging
        if rank == 0 and (step % args.log_interval == 0 or step == args.total_steps - 1):
            now = time.time()
            elapsed_hours = (now - start_wall_time) / 3600.0
            est_cost = elapsed_hours * V100_4X_HOURLY_COST
            print(
                f"[Step {step:05d}/{args.total_steps}] "
                f"Loss: {step_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"MFU: {mfu * 100:.1f}% | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"Dead: {telemetry['dead_frac'] * 100:.1f}% | "
                f"Imb: {telemetry['imbalance']:.2f} | "
                f"GradNorm: {grad_norm:.2f} | "
                f"Cost: ${est_cost:.2f}"
            )
            last_log_time = now

        # Periodic Checkpoint saving
        if ((step > 0 and step % args.save_interval == 0) or step == args.total_steps - 1):
            elapsed_hours = (time.time() - start_wall_time) / 3600.0
            meta_info = {
                "step": step,
                "total_tokens_seen": total_tokens_seen,
                "loss": step_loss,
                "mfu": mfu,
                "cost": elapsed_hours * V100_4X_HOURLY_COST,
            }
            if val_loader is not None and step % args.validation_interval == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = val_loader.next_batch(device)
                    meta_info["val_loss"] = model(vx, labels=vy)["loss"].item()
                model.train()
            saved_path = ckpt_manager.save(
                step=step,
                model=raw_model,
                optimizer=optimizer,
                spike_guard=spike_guard,
                data_loader_state=data_loader.state_dict(),
                extra_meta=meta_info,
                rank=rank,
                world_size=(dist.get_world_size() if distributed else 1),
            )
            if rank == 0: print(f"[*] Checkpoint saved at step {step}: {saved_path}")

    if distributed: dist.barrier(); dist.destroy_process_group()
    if rank == 0:
        print("\n" + "=" * 80); print(f" Training Complete! Total tokens: {total_tokens_seen:,}"); print("=" * 80)


if __name__ == "__main__":
    main()
