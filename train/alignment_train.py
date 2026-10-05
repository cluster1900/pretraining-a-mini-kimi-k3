"""
Unified Alignment & Post-Training Entry Point for Mini Kimi K3.
Supports the post-training stages in TRAINING_PLAN.md:
1. SFT:  supervised fine-tuning on audited ``input_ids``/``labels`` rows; loss is
         the model's own chunked loss (``compute_logits=False``, lm + 0.3*MTP)
2. RM:   reward model (Bradley-Terry) on chosen/rejected preference pairs
3. DPO:  frozen reference; summed response log-probs by default,
         ``--dpo_length_norm`` for per-token means; logs implicit rewards/margins
4. PPO:  policy/reference/reward/value, token-level GAE, per-token k3 KL,
         ``--ppo_epochs`` passes over each rollout (see ``rl_trainer.py``)
5. GRPO: group-relative updates with format + gold-answer rule rewards, no
         value model. Prompts are the prompt part of audited SFT rows (tokens
         before the first supervised label); gold answers are joined from the
         audited decontaminated stage (``alignment_readiness``)

``--steps`` counts micro-batches (records) for SFT/RM/DPO, grouped into
optimizer updates of ``--grad_accum_steps``; the last partial group is averaged
over its actual size. For PPO/GRPO ``--steps`` counts rollout iterations.
Logging happens every ``--log_interval`` optimizer updates.

Outputs go to ``/data/mini-k3/checkpoints/alignment-<mode>`` by default. A
directory that already holds model artifacts is refused unless ``--overwrite``;
a pretraining checkpoint directory is always refused. ``alignment_meta.json``
records mode, base checkpoint, sequence length, dataset evidence and arguments.

Precision: FP16 autocast + GradScaler for trainable models on CUDA, frozen
reference/reward models in FP16 (``--frozen_dtype``) on ``--ref_device`` /
``--reward_device``. The value model may sit on ``--value_device``.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

DEFAULT_OUTPUT_ROOT = "/data/mini-k3/checkpoints"
ARTIFACTS = {
    "sft": ("model.pt",),
    "rm": ("reward_model.pt",),
    "dpo": ("model.pt",),
    "ppo": ("model.pt", "value_model.pt"),
    "grpo": ("model.pt",),
}
KNOWN_ARTIFACTS = ("model.pt", "reward_model.pt", "value_model.pt", "alignment_meta.json")
DEFAULT_KL = {"ppo": 0.02, "grpo": 0.04}


# ---------------------------------------------------------------------------
# Output directory policy
# ---------------------------------------------------------------------------

def default_output_dir(mode: str) -> Path:
    return Path(DEFAULT_OUTPUT_ROOT) / f"alignment-{mode}"


def check_output_dir(out_dir, mode: str, overwrite: bool, input_paths=()) -> Path:
    """Refuse to clobber another run's artifacts or a pretraining checkpoint."""
    out_dir = Path(out_dir).resolve()
    for p in input_paths:
        if p is None:
            continue
        p = Path(p).resolve()
        source_dir = p if p.is_dir() else p.parent
        if out_dir == source_dir:
            raise ValueError(f"--output_dir {out_dir} is an input checkpoint directory")
    if not out_dir.exists():
        return out_dir
    if (out_dir / "meta.pt").exists() or (out_dir / "COMPLETE").exists() or any(out_dir.glob("step_*")):
        raise ValueError(f"{out_dir} holds a pretraining checkpoint; choose another --output_dir")
    present = [name for name in KNOWN_ARTIFACTS if (out_dir / name).exists()]
    if not present:
        return out_dir
    previous = None
    meta_file = out_dir / "alignment_meta.json"
    if meta_file.is_file():
        try:
            previous = json.loads(meta_file.read_text(encoding="utf-8")).get("mode")
        except Exception:
            previous = "unreadable"
    if not overwrite:
        what = f"artifacts of mode {previous!r}" if previous else "existing artifacts"
        raise ValueError(f"{out_dir} already contains {what} ({', '.join(present)}); "
                         "pass --overwrite or choose another --output_dir")
    return out_dir


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_save(state, path: Path) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_alignment_meta(out_dir: Path, mode: str, payload: dict) -> None:
    meta = dict(payload)
    meta["mode"] = mode
    meta["artifacts"] = {name: _sha256(out_dir / name) for name in ARTIFACTS[mode]}
    meta["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = out_dir / "alignment_meta.json.incomplete"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_dir / "alignment_meta.json")


# ---------------------------------------------------------------------------
# Generic accumulated optimisation loop (SFT / RM / DPO)
# ---------------------------------------------------------------------------

def run_accumulated(model, optimizer, scaler, balancer, steps, accum, micro_fn, tag,
                    log_interval=10, max_grad_norm=1.0, log=print):
    """``steps`` micro-batches in groups of ``accum``; each group is one update.

    ``micro_fn()`` returns ``(loss, metrics)``; the loss is divided by the
    group's actual size (the last group may be shorter) and metrics are
    averaged over the group. Returns the list of per-update metric dicts.
    """
    from train.rl_trainer import optimizer_update
    if steps < 1 or accum < 1:
        raise ValueError("steps and grad_accum_steps must be >= 1")
    done = 0
    update = 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    while done < steps:
        group = min(accum, steps - done)
        sums = {}
        for _ in range(group):
            loss, metrics = micro_fn()
            scaler.scale(loss / group).backward()
            metrics = dict(metrics, loss=float(loss.detach()))
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            done += 1
        grad_norm = optimizer_update(model, optimizer, scaler, max_grad_norm, balancer)
        update += 1
        averaged = {key: value / group for key, value in sums.items()}
        averaged.update(update=update, micro_steps=done, grad_norm=grad_norm)
        history.append(averaged)
        if update % log_interval == 0 or done == steps or update == 1:
            body = " | ".join(f"{k}={v:+.4f}" for k, v in averaged.items()
                              if k not in ("update", "micro_steps"))
            log(f"[{tag} update {update:05d} | micro {done}/{steps}] {body}")
    return history


# ---------------------------------------------------------------------------
# Data streams
# ---------------------------------------------------------------------------

def record_stream(path):
    while True:
        produced = False
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    produced = True
                    yield json.loads(line)
        if not produced:
            raise ValueError(f"No records in {path}")


def prompt_stream(path, kind, max_prompt_len, golds=None):
    """Yield ``(prompt_ids, gold)``; never includes a gold response in the prompt.

    preference rows → ``prompt_ids`` (fallback ``chosen_ids[:prompt_len]``);
    sft rows → tokens before the first supervised label (``sft_prompt``).
    With ``golds`` only rows whose id has an audited answer are used.
    """
    from train.alignment_fit import sft_prompt
    usable = 0
    for seen, row in enumerate(record_stream(path), start=1):
        if kind == "preference":
            prompt = row.get("prompt_ids")
            if prompt is None and row.get("prompt_len"):
                prompt = row["chosen_ids"][: row["prompt_len"]]
        else:
            prompt = sft_prompt(row["input_ids"], row["labels"])
        if not prompt or len(prompt) > max_prompt_len:
            prompt = None
        gold = None
        if golds is not None:
            gold = golds.get(row.get("id"))
            if gold is None:
                prompt = None
        if prompt is None:
            if seen > 100_000 and usable == 0:
                raise ValueError(f"no usable prompt in the first {seen} rows of {path}")
            continue
        usable += 1
        yield list(prompt), gold


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="Mini K3 Unified Post-Training Entry Point")
    parser.add_argument("--mode", choices=("sft", "rm", "dpo", "ppo", "grpo"), required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Causal LM: run root (newest complete step_*), step dir, alignment dir or model.pt")
    parser.add_argument("--jsonl", required=True, help="Audited dataset JSONL")
    parser.add_argument("--alignment-manifest",
                        default="/data/mini-k3/data/prepared-v2-supplement-v2/manifests/alignment.json",
                        help="Audited alignment manifest that must bind --jsonl")
    parser.add_argument("--steps", type=int, default=1000,
                        help="SFT/RM/DPO: micro-batches; PPO/GRPO: rollout iterations")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Micro-batches per optimizer update")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N optimizer updates")
    parser.add_argument("--output_dir", default=None, help="Default: /data/mini-k3/checkpoints/alignment-<mode>")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing artifacts in --output_dir")
    parser.add_argument("--sequence_length", type=int, default=None,
                        help="Token limit per sequence. Default: checkpoint's trained length, else 2048")
    parser.add_argument("--seed", type=int, default=0)
    # Devices / precision
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device of the trainable policy / SFT / RM model")
    parser.add_argument("--ref_device", default=None, help="Frozen reference device (default --device)")
    parser.add_argument("--reward_device", default=None, help="Frozen reward model device (default --device)")
    parser.add_argument("--value_device", default=None, help="PPO value model device (default --device)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="FP16 autocast + GradScaler on CUDA")
    parser.add_argument("--frozen_dtype", choices=("fp16", "fp32"), default="fp16",
                        help="Weights dtype of frozen reference/reward models on CUDA")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Sequences per forward in PPO/GRPO rollout scoring and updates")
    # RM / PPO inputs
    parser.add_argument("--reward_checkpoint", default=None,
                        help="PPO: directory with reward_model.pt or the file itself")
    parser.add_argument("--value_checkpoint", default=None,
                        help="PPO: directory with value_model.pt; default initialises from --checkpoint")
    # DPO
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_length_norm", action="store_true",
                        help="Use per-token mean response log-probs instead of sums")
    # RL
    parser.add_argument("--tokenizer_model", default=None, help="Tokenizer directory. Required for GRPO.")
    parser.add_argument("--gold_marker", default=None,
                        help="GRPO: override the audited decontaminated COMPLETE.json used for gold answers")
    parser.add_argument("--group_size", type=int, default=4, help="Completions per prompt in GRPO")
    parser.add_argument("--prompt_batch", type=int, default=4, help="Prompts per PPO/GRPO iteration")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Rollout response limit")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=None, help="Default 0.02 (PPO) / 0.04 (GRPO)")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=2, help="PPO passes over each rollout")
    parser.add_argument("--grpo_epochs", type=int, default=1, help="GRPO passes over each rollout")
    parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE discount")
    parser.add_argument("--lam", type=float, default=0.95, help="PPO GAE lambda")
    parser.add_argument("--eos_token_id", type=int, default=163585, help="EOS id (K3 tokenizer [EOS])")
    parser.add_argument("--pad_token_id", type=int, default=163839, help="PAD id for rollout padding")
    parser.add_argument("--no_balancer", action="store_true", help="Do not run the MoE balancer after updates")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from train.config import DEFAULT_CONFIG
    from train.alignment_readiness import check_alignment_jsonl, load_audited_gold_answers
    from train.alignment_models import (
        RewardModel, ValueModel, config_for_checkpoint, load_causal_model, resolve_checkpoint,
    )
    from train.alignment import (
        dpo_loss, dpo_metrics, k3_kl, masked_mean, pairwise_reward_loss, response_logprob, token_logprobs,
    )
    from train.alignment_fit import fit_preference, fit_sft
    from train.engine.muon import build_optimizer
    from train.rl_trainer import (
        GRPOTrainer, PPOTrainer, amp_autocast, make_balancer, make_grad_scaler, model_device, prepare_frozen,
    )

    torch.manual_seed(args.seed)
    evidence = check_alignment_jsonl(args.alignment_manifest, args.jsonl, args.mode, DEFAULT_CONFIG.vocab_size)

    device = torch.device(args.device)
    ref_device = torch.device(args.ref_device or args.device)
    reward_device = torch.device(args.reward_device or args.device)
    value_device = torch.device(args.value_device or args.device)
    frozen_fp16 = args.frozen_dtype == "fp16"
    use_amp = bool(args.amp)

    info = resolve_checkpoint(args.checkpoint)
    cfg = config_for_checkpoint(info)
    limit = args.sequence_length or info.sequence_length or DEFAULT_CONFIG.sequence_length
    if info.sequence_length is not None and limit > info.sequence_length:
        print(f"[!] --sequence_length {limit} exceeds the checkpoint's trained length {info.sequence_length}")
    cfg.sequence_length = limit
    if args.mode in ("ppo", "grpo") and limit - args.max_new_tokens < 2:
        raise ValueError(f"--max_new_tokens {args.max_new_tokens} leaves no prompt room in sequence length {limit}")

    out_dir = check_output_dir(args.output_dir or default_output_dir(args.mode), args.mode, args.overwrite,
                               (info.directory, args.reward_checkpoint, args.value_checkpoint))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_checkpoint": str(info.model_file), "sequence_length": limit,
        "dataset": evidence,
        "args": {k: v for k, v in vars(args).items()},
    }
    balance = not args.no_balancer

    print("=" * 80)
    print(f" Mini Kimi K3 Alignment Engine: Mode = [{args.mode.upper()}]")
    print(f" Checkpoint: {info.model_file}")
    print(f" Dataset:    {args.jsonl} ({evidence['kind']}, sha256 {evidence['jsonl_sha256'][:16]}...)")
    print(f" Output:     {out_dir}")
    print(f" Sequence:   {limit} | Steps {args.steps} | LR {args.lr:.2e} | Accum {args.grad_accum_steps}")
    print(f" Devices:    policy {device} | ref {ref_device} | reward {reward_device} | value {value_device}"
          f" | amp {use_amp and device.type == 'cuda'} | frozen {args.frozen_dtype}")
    print("=" * 80)

    def new_optimizer(model):
        return build_optimizer(
            (q for q in model.parameters() if q.requires_grad),
            lr=args.lr,
            weight_decay=0.01,
            muon_update_scale=cfg.muon_update_scale,
        )

    def next_fitted(stream, fit, what):
        for _ in range(10000):
            fitted = fit(next(stream))
            if fitted is not None:
                return fitted
        raise ValueError(f"No {what} row fits within sequence length {limit}")

    # ---------------------------------------------------------------- SFT
    if args.mode == "sft":
        model, _, _ = load_causal_model(args.checkpoint, device, config=cfg)
        model.train()
        opt = new_optimizer(model)
        scaler = make_grad_scaler(device, use_amp)
        stream = record_stream(args.jsonl)

        def micro():
            ids, labels, _ = next_fitted(stream, lambda r: fit_sft(r["input_ids"], r["labels"], limit), "SFT")
            ids_t = torch.tensor([ids], device=device)
            labels_t = torch.tensor([labels], device=device)
            with amp_autocast(device, use_amp):
                out = model(ids_t, labels=labels_t, compute_logits=False)
            metrics = {"lm_loss": float(out["lm_loss"].detach())}
            if out.get("mtp_loss") is not None:
                metrics["mtp_loss"] = float(out["mtp_loss"].detach())
            return out["loss"].float(), metrics

        run_accumulated(model, opt, scaler, make_balancer(model, balance), args.steps,
                        args.grad_accum_steps, micro, "SFT", args.log_interval)
        atomic_save(model.state_dict(), out_dir / "model.pt")

    # ---------------------------------------------------------------- RM
    elif args.mode == "rm":
        rm_model = RewardModel(cfg, checkpoint=args.checkpoint, source="causal").to(device).train()
        opt = new_optimizer(rm_model)
        scaler = make_grad_scaler(device, use_amp)
        stream = record_stream(args.jsonl)

        def micro():
            c_ids, r_ids, _ = next_fitted(
                stream, lambda r: fit_preference(r["chosen_ids"], r["rejected_ids"], r.get("prompt_len", 0), limit),
                "preference")
            with amp_autocast(device, use_amp):
                r_chosen = rm_model(torch.tensor([c_ids], device=device))
                r_rejected = rm_model(torch.tensor([r_ids], device=device))
            loss = pairwise_reward_loss(r_chosen.float(), r_rejected.float())
            margin = (r_chosen - r_rejected).detach().float()
            return loss, {"margin": float(margin.mean()), "acc": float((margin > 0).float().mean())}

        run_accumulated(rm_model, opt, scaler, make_balancer(rm_model, balance), args.steps,
                        args.grad_accum_steps, micro, "RM", args.log_interval)
        atomic_save(rm_model.state_dict(), out_dir / "reward_model.pt")

    # ---------------------------------------------------------------- DPO
    elif args.mode == "dpo":
        model, _, _ = load_causal_model(args.checkpoint, device, config=cfg)
        model.train()
        ref, _, _ = load_causal_model(args.checkpoint, None, config=cfg)
        ref = prepare_frozen(ref, ref_device, frozen_fp16)
        opt = new_optimizer(model)
        scaler = make_grad_scaler(device, use_amp)
        stream = record_stream(args.jsonl)

        def micro():
            c_list, r_list, prompt_len = next_fitted(
                stream, lambda r: fit_preference(r["chosen_ids"], r["rejected_ids"], r.get("prompt_len", 0), limit),
                "preference")
            logps = {}
            for name, seq in (("c", c_list), ("r", r_list)):
                ids = torch.tensor([seq], device=device)
                mask = torch.zeros((1, len(seq) - 1), device=device)
                mask[:, prompt_len - 1:] = 1.0
                with amp_autocast(device, use_amp):
                    pol = token_logprobs(model, ids).float()
                with torch.no_grad(), amp_autocast(ref_device, use_amp):
                    rf = token_logprobs(ref, ids.to(ref_device)).float().to(device)
                logps[name] = (pol, rf, mask)
            seq_lp = {name: (response_logprob(p, m, args.dpo_length_norm),
                             response_logprob(r, m, args.dpo_length_norm)) for name, (p, r, m) in logps.items()}
            loss = dpo_loss(seq_lp["c"][0], seq_lp["r"][0], seq_lp["c"][1], seq_lp["r"][1], beta=args.dpo_beta)
            metrics = dpo_metrics(seq_lp["c"][0], seq_lp["r"][0], seq_lp["c"][1], seq_lp["r"][1], args.dpo_beta)
            pol_c, ref_c, mask_c = logps["c"]
            metrics["kl_chosen_k3"] = float(masked_mean(k3_kl(pol_c.detach(), ref_c), mask_c))
            return loss, metrics

        run_accumulated(model, opt, scaler, make_balancer(model, balance), args.steps,
                        args.grad_accum_steps, micro, "DPO", args.log_interval)
        atomic_save(model.state_dict(), out_dir / "model.pt")

    # ---------------------------------------------------------------- PPO
    elif args.mode == "ppo":
        if not args.reward_checkpoint:
            raise ValueError("--reward_checkpoint is required for PPO; a causal LM checkpoint is not a reward model")
        policy, _, _ = load_causal_model(args.checkpoint, device, config=cfg)
        policy.train()
        ref, _, _ = load_causal_model(args.checkpoint, None, config=cfg)
        ref = prepare_frozen(ref, ref_device, frozen_fp16)
        reward_model = prepare_frozen(RewardModel(cfg, checkpoint=args.reward_checkpoint, source="scalar"),
                                      reward_device, frozen_fp16)
        if args.value_checkpoint:
            value_model = ValueModel(cfg, checkpoint=args.value_checkpoint, source="scalar")
        else:
            value_model = ValueModel(cfg, checkpoint=args.checkpoint, source="causal")
        value_model = value_model.to(value_device).train()
        trainer = PPOTrainer(
            policy_model=policy, ref_model=ref, reward_model=reward_model, value_model=value_model,
            policy_optimizer=new_optimizer(policy), value_optimizer=build_optimizer(
                (q for q in value_model.parameters() if q.requires_grad), lr=args.lr * 2, weight_decay=0.01),
            clip_eps=args.clip_eps, kl_beta=args.kl_beta if args.kl_beta is not None else DEFAULT_KL["ppo"],
            gamma=args.gamma, lam=args.lam, ppo_epochs=args.ppo_epochs,
            micro_batch_size=args.micro_batch_size, use_amp=use_amp, pad_token_id=args.pad_token_id,
            balance=balance,
        )
        prompts = prompt_stream(args.jsonl, evidence["kind"], limit - args.max_new_tokens)
        for step in range(args.steps):
            batch = [next(prompts)[0] for _ in range(args.prompt_batch)]
            rollout = trainer.generate_rollout(batch, max_new_tokens=args.max_new_tokens,
                                               eos_token_id=args.eos_token_id,
                                               temperature=args.temperature, top_p=args.top_p)
            metrics = trainer.train_step(rollout)
            if (step + 1) % args.log_interval == 0 or step == 0 or step + 1 == args.steps:
                print(f"[PPO iter {step + 1:05d}/{args.steps}] " +
                      " | ".join(f"{k}={v:+.4f}" for k, v in metrics.items()))
        atomic_save(policy.state_dict(), out_dir / "model.pt")
        atomic_save(value_model.state_dict(), out_dir / "value_model.pt")
        meta["reward_checkpoint"] = str(args.reward_checkpoint)

    # ---------------------------------------------------------------- GRPO
    elif args.mode == "grpo":
        if not args.tokenizer_model:
            raise ValueError("--tokenizer_model is required for GRPO so completions can be checked")
        if args.prompt_batch < 1 or args.group_size < 2:
            raise ValueError("GRPO needs at least one prompt and two samples in each group")
        golds, gold_report = load_audited_gold_answers(evidence, decontaminated_marker=args.gold_marker)
        print(f"[*] Gold answers: {gold_report['with_answer']} of {gold_report['train_rows']} train rows "
              f"(marker sha256 {gold_report['marker_sha256'][:16]}...)")
        meta["gold_answers"] = gold_report
        from train.data.tokenizer import K3Tokenizer
        tokenizer = K3Tokenizer(args.tokenizer_model)
        if tokenizer.fingerprint != evidence["tokenizer_fingerprint"]:
            raise ValueError("tokenizer fingerprint differs from the audited alignment manifest")
        policy, _, _ = load_causal_model(args.checkpoint, device, config=cfg)
        policy.train()
        ref, _, _ = load_causal_model(args.checkpoint, None, config=cfg)
        ref = prepare_frozen(ref, ref_device, frozen_fp16)
        trainer = GRPOTrainer(
            policy, ref, new_optimizer(policy), group_size=args.group_size, clip_eps=args.clip_eps,
            kl_beta=args.kl_beta if args.kl_beta is not None else DEFAULT_KL["grpo"],
            epochs=args.grpo_epochs, micro_batch_size=args.micro_batch_size, use_amp=use_amp,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, balance=balance,
        )
        prompts = prompt_stream(args.jsonl, evidence["kind"], limit - args.max_new_tokens, golds=golds)
        for step in range(args.steps):
            batch = [next(prompts) for _ in range(args.prompt_batch)]
            metrics = trainer.step([p for p, _ in batch], [g for _, g in batch], tokenizer,
                                   max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                   top_p=args.top_p)
            if (step + 1) % args.log_interval == 0 or step == 0 or step + 1 == args.steps:
                print(f"[GRPO iter {step + 1:05d}/{args.steps}] " +
                      " | ".join(f"{k}={v:+.4f}" for k, v in metrics.items()))
        atomic_save(policy.state_dict(), out_dir / "model.pt")

    write_alignment_meta(out_dir, args.mode, meta)
    print(f"[*] {args.mode.upper()} finished. Artifacts: "
          + ", ".join(str(out_dir / name) for name in ARTIFACTS[args.mode]))


if __name__ == "__main__":
    main()
