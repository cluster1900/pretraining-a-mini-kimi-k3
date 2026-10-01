"""
Unified Alignment & Post-Training Entry Point for Mini Kimi K3.
Supports the post-training stages in TRAINING_PLAN.md:
1. SFT: Supervised Fine-Tuning on packed conversation JSONL
2. RM:  Reward Model training on chosen/rejected preference pairs
3. DPO: Direct Preference Optimization with frozen reference model
4. PPO: Online Reinforcement Learning with Actor-Critic, GAE, and KL control
5. GRPO: group-relative updates with format and answer rewards, no value model
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from train.config import DEFAULT_CONFIG
from train.models.mini_k3 import MiniK3ForCausalLM
from train.alignment_models import RewardModel, ValueModel
from train.alignment import dpo_loss, sequence_logprob, pairwise_reward_loss
from train.alignment_fit import fit_preference, fit_sft
from train.rl_trainer import GRPOTrainer, PPOTrainer
from train.engine.muon import build_optimizer
from train.alignment_readiness import check_alignment_jsonl


def load_causal_model(checkpoint: str, device: torch.device) -> MiniK3ForCausalLM:
    model = MiniK3ForCausalLM(DEFAULT_CONFIG).to(device)
    ckpt_p = Path(checkpoint)
    model_file = ckpt_p / "model.pt" if ckpt_p.is_dir() else ckpt_p
    state = torch.load(model_file, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Mini K3 Unified Post-Training Entry Point")
    parser.add_argument("--mode", choices=("sft", "rm", "dpo", "ppo", "grpo"), required=True, help="Alignment training mode")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory or model.pt")
    parser.add_argument("--jsonl", required=True, help="Path to dataset JSONL")
    parser.add_argument(
        "--alignment-manifest",
        default="/data/mini-k3/data/prepared-v2-supplement-v2/manifests/alignment.json",
        help="Audited alignment manifest that must bind --jsonl",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="/data/mini-k3/checkpoints/alignment", help="Directory to save checkpoint")
    parser.add_argument("--reward_checkpoint", type=str, default=None, help="Path to trained reward model (required for PPO)")
    parser.add_argument("--tokenizer_model", default=None, help="Tokenizer directory. Required for GRPO.")
    parser.add_argument("--group_size", type=int, default=4, help="Completions sampled per prompt in GRPO.")
    parser.add_argument("--prompt_batch", type=int, default=4, help="Prompts per GRPO step. Each is sampled group_size times.")
    args = parser.parse_args()

    alignment_evidence = check_alignment_jsonl(
        args.alignment_manifest, args.jsonl, args.mode, DEFAULT_CONFIG.vocab_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def records():
        while True:
            produced = False
            with open(args.jsonl, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        produced = True
                        yield json.loads(line)
            if not produced:
                raise ValueError(f"No records in {args.jsonl}")

    stream = records()

    print("=" * 80)
    print(f" Mini Kimi K3 Alignment Engine: Mode = [{args.mode.upper()}]")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Dataset:    {args.jsonl}")
    print(f" Dataset audit: {alignment_evidence['jsonl_sha256']}")
    print(f" Steps:      {args.steps} | LR = {args.lr:.2e} | Accum = {args.grad_accum_steps}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Mode 1: SFT (Supervised Fine-Tuning)
    # -------------------------------------------------------------------------
    if args.mode == "sft":
        model = load_causal_model(args.checkpoint, device).train()
        opt = build_optimizer((q for q in model.parameters() if q.requires_grad), lr=args.lr, weight_decay=0.01)
        opt.zero_grad()
        accum_loss = 0.0

        limit = DEFAULT_CONFIG.sequence_length
        for step in range(args.steps):
            fitted = None
            for _ in range(10000):
                row = next(stream)
                fitted = fit_sft(row["input_ids"], row["labels"], limit)
                if fitted is not None:
                    break
            if fitted is None:
                raise ValueError("No SFT row retains a supervised token within the sequence limit")
            ids_list, labels_list, _dropped = fitted
            ids = torch.tensor([ids_list], device=device)
            labels = torch.tensor([labels_list], device=device)
            out = model(ids, labels=labels)
            loss = out["loss"] / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or step == args.steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if step % 10 == 0:
                    print(f"[SFT Step {step:04d}/{args.steps}] Loss = {accum_loss / args.grad_accum_steps:.4f}")
                accum_loss = 0.0

        torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"[*] SFT training finished. Model saved to: {out_dir / 'model.pt'}")

    # -------------------------------------------------------------------------
    # Mode 2: RM (Reward Model Training)
    # -------------------------------------------------------------------------
    elif args.mode == "rm":
        rm_model = RewardModel(DEFAULT_CONFIG, checkpoint=args.checkpoint).to(device).train()
        opt = build_optimizer((q for q in rm_model.parameters() if q.requires_grad), lr=args.lr, weight_decay=0.01)
        opt.zero_grad()
        accum_loss = 0.0

        limit = DEFAULT_CONFIG.sequence_length
        for step in range(args.steps):
            fitted = None
            for _ in range(10000):
                row = next(stream)
                fitted = fit_preference(row["chosen_ids"], row["rejected_ids"], row.get("prompt_len", 0), limit)
                if fitted is not None:
                    break
            if fitted is None:
                raise ValueError("No preference row retains an answer within the sequence limit")
            chosen_ids, rejected_ids, _prompt_len = fitted
            c = torch.tensor([chosen_ids], device=device)
            r = torch.tensor([rejected_ids], device=device)

            r_chosen = rm_model(c)
            r_rejected = rm_model(r)

            loss = pairwise_reward_loss(r_chosen, r_rejected) / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or step == args.steps - 1:
                torch.nn.utils.clip_grad_norm_(rm_model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if step % 10 == 0:
                    margin = (r_chosen - r_rejected).mean().item()
                    print(f"[RM Step {step:04d}/{args.steps}] Loss = {accum_loss / args.grad_accum_steps:.4f} | Margin = {margin:+.4f}")
                accum_loss = 0.0

        torch.save(rm_model.state_dict(), out_dir / "reward_model.pt")
        print(f"[*] Reward Model training finished. Saved to: {out_dir / 'reward_model.pt'}")

    # -------------------------------------------------------------------------
    # Mode 3: DPO (Direct Preference Optimization)
    # -------------------------------------------------------------------------
    elif args.mode == "dpo":
        model = load_causal_model(args.checkpoint, device).train()
        ref = load_causal_model(args.checkpoint, device).eval()
        for q in ref.parameters():
            q.requires_grad = False

        opt = build_optimizer((q for q in model.parameters() if q.requires_grad), lr=args.lr, weight_decay=0.01)
        opt.zero_grad()
        accum_loss = 0.0

        limit = DEFAULT_CONFIG.sequence_length
        for step in range(args.steps):
            fitted = None
            for _ in range(10000):
                row = next(stream)
                prompt_len = row.get("prompt_len", 0)
                if not prompt_len:
                    for t1, t2 in zip(row["chosen_ids"], row["rejected_ids"]):
                        if t1 == t2:
                            prompt_len += 1
                        else:
                            break
                fitted = fit_preference(row["chosen_ids"], row["rejected_ids"], prompt_len, limit)
                if fitted is not None:
                    break
            if fitted is None:
                raise ValueError("No preference row retains an answer within the sequence limit")
            c_list, r_list, prompt_len = fitted
            c = torch.tensor([c_list], device=device)
            r = torch.tensor([r_list], device=device)

            c_labels = c.clone()
            r_labels = r.clone()
            c_labels[:, :prompt_len] = -100
            r_labels[:, :prompt_len] = -100

            with torch.no_grad():
                rc = ref(c)["logits"]
                rr = ref(r)["logits"]
            pc = model(c)["logits"]
            pr = model(r)["logits"]

            loss = dpo_loss(
                sequence_logprob(pc, c_labels, average_logprobs=False),
                sequence_logprob(pr, r_labels, average_logprobs=False),
                sequence_logprob(rc, c_labels, average_logprobs=False),
                sequence_logprob(rr, r_labels, average_logprobs=False),
            ) / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or step == args.steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if step % 10 == 0:
                    print(f"[DPO Step {step:04d}/{args.steps}] Loss = {accum_loss / args.grad_accum_steps:.4f}")
                accum_loss = 0.0

        torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"[*] DPO training finished. Model saved to: {out_dir / 'model.pt'}")

    # -------------------------------------------------------------------------
    # Mode 4: PPO (Online Reinforcement Learning)
    # -------------------------------------------------------------------------
    elif args.mode == "ppo":
        if not args.reward_checkpoint:
            raise ValueError("--reward_checkpoint is required for PPO; a causal LM checkpoint is not a reward model")
        policy = load_causal_model(args.checkpoint, device).train()
        ref = load_causal_model(args.checkpoint, device).eval()

        rm_ckpt = args.reward_checkpoint
        reward_model = RewardModel(DEFAULT_CONFIG, checkpoint=rm_ckpt).to(device).eval()
        value_model = ValueModel(DEFAULT_CONFIG, checkpoint=args.checkpoint).to(device).train()

        policy_opt = build_optimizer((q for q in policy.parameters() if q.requires_grad), lr=args.lr, weight_decay=0.01)
        value_opt = build_optimizer((q for q in value_model.parameters() if q.requires_grad), lr=args.lr * 2, weight_decay=0.01)

        ppo_trainer = PPOTrainer(
            policy_model=policy,
            ref_model=ref,
            reward_model=reward_model,
            value_model=value_model,
            policy_optimizer=policy_opt,
            value_optimizer=value_opt,
        )

        batch_prompts_size = 4
        for step in range(args.steps):
            batch_rows = [next(stream) for _ in range(batch_prompts_size)]
            prompts = [r.get("prompt_ids", r.get("input_ids", [163584])) for r in batch_rows]

            # 1. Rollout with frozen reward model and baseline value model
            rollout = ppo_trainer.generate_rollout(prompts, max_new_tokens=64, eos_token_id=DEFAULT_CONFIG.vocab_size - 255)

            # 2. PPO Update Step
            metrics = ppo_trainer.train_step(rollout)

            if step % 5 == 0:
                print(
                    f"[PPO Step {step:04d}/{args.steps}] PolicyLoss = {metrics['policy_loss']:.4f} | "
                    f"ValueLoss = {metrics['value_loss']:.4f} | KL = {metrics['kl']:.4f} | "
                    f"MeanReward = {metrics['reward_mean']:+.4f}"
                )

        torch.save(policy.state_dict(), out_dir / "model.pt")
        print(f"[*] PPO training finished. Policy checkpoint saved to: {out_dir / 'model.pt'}")

    elif args.mode == "grpo":
        if not args.tokenizer_model:
            raise ValueError("--tokenizer_model is required for GRPO so completions can be checked")
        if args.prompt_batch < 1 or args.group_size < 2:
            raise ValueError("GRPO needs at least one prompt and two samples in each group")
        from train.data.tokenizer import K3Tokenizer
        tokenizer = K3Tokenizer(args.tokenizer_model)
        policy = load_causal_model(args.checkpoint, device).train()
        ref = load_causal_model(args.checkpoint, device).eval()
        opt = build_optimizer((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
        trainer = GRPOTrainer(policy, ref, opt, group_size=args.group_size)
        for step in range(args.steps):
            batch_rows = [next(stream) for _ in range(args.prompt_batch)]
            prompts = [row.get("prompt_ids", row.get("input_ids", [1])) for row in batch_rows]
            golds = [row.get("answer") for row in batch_rows]
            metrics = trainer.step(prompts, golds, tokenizer)
            if step % 5 == 0:
                print(
                    f"[GRPO Step {step:04d}/{args.steps}] Loss = {metrics['loss']:.4f} | "
                    f"Reward = {metrics['reward_mean']:+.3f} | KL = {metrics['kl']:.4f}"
                )
        torch.save(policy.state_dict(), out_dir / "model.pt")
        print(f"[*] GRPO training finished. Policy checkpoint saved to: {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
