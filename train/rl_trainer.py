"""
PPO / GRPO orchestration for Mini Kimi K3 (TRAINING_PLAN.md sections 5-6).

PPO (actor-critic):
- Rollouts are sampled with the KV cache and cut at the first EOS (EOS kept,
  everything after it dropped). Rollouts record per-token old log-probs,
  reference log-probs and value predictions ONCE, at rollout time.
- Rewards: the frozen ``RewardModel`` score of prompt+response (pooled at the
  last real token, i.e. the EOS, exactly like RM training) is placed on the
  last response token. Advantages are token-level GAE(gamma, lambda) over the
  per-token values of ``ValueModel(per_token=True)``, then whitened over
  response tokens. Returns = advantages + values.
- Loss: token-level clipped surrogate + ``kl_beta`` * per-token k3 KL
  ``exp(ref-new) - (ref-new) - 1`` against the frozen reference (a loss term,
  not folded into the reward), plus a clipped value loss. ``ppo_epochs``
  (default 2) passes over the same rollout, so the ratio and clipping are
  meaningful from the second pass on.

GRPO: no value model. ``group_size`` samples per prompt, rewards from
``rule_reward`` (format + gold answer), group-normalised sequence advantages
broadcast to every response token, token-level clipped surrogate and k3 KL,
per-sequence token mean. Old log-probs are captured at rollout time and
``epochs`` inner passes are allowed (default 1 = on-policy).

Memory plan for the 1.15B model on 32 GB V100s (FP16, no BF16):
- trainable models (policy, value) keep FP32 master weights, run forward under
  ``torch.autocast(float16)`` and backward through ``torch.amp.GradScaler``.
  Each trainable model needs about 4.6 GB weights + 4.6 GB grads + optimizer
  state (Muon momentum / AdamW moments, ~5-9 GB) + activations, i.e. roughly
  16-20 GB, so policy and value should sit on different GPUs.
- frozen models (reference, reward) are cast to FP16 (``.half()``, ~2.3 GB
  each) and only run inference.
- Suggested placement on 4 GPUs: policy cuda:0, value cuda:1, reference
  cuda:2, reward cuda:3 (``--value_device/--ref_device/--reward_device``).
  Tensors are moved between devices per micro-batch; results return to the
  policy device.
- log-probs are computed from hidden states in chunks (``token_logprobs``), so
  full-vocabulary logits are never resident.
On CPU everything stays FP32 and autocast/GradScaler are disabled.
"""

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from train.alignment import (
    clipped_policy_loss, k3_kl, masked_mean, token_clipped_surrogate, token_logprobs,
)


# ---------------------------------------------------------------------------
# Device / precision helpers
# ---------------------------------------------------------------------------

def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def amp_autocast(device: torch.device, enabled: bool = True):
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool = True):
    on = bool(enabled and torch.device(device).type == "cuda")
    return torch.amp.GradScaler("cuda", enabled=on)


def prepare_frozen(model: nn.Module, device, fp16: bool = True) -> nn.Module:
    """Move a frozen model to ``device``; FP16 weights on CUDA when ``fp16``."""
    device = torch.device(device)
    model = model.to(device)
    if fp16 and device.type == "cuda":
        model = model.half()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def optimizer_update(model, optimizer, scaler, max_grad_norm, balancer=None) -> float:
    """Unscale, clip, step, update the scaler, zero grads, then rebalance MoE routing."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if balancer is not None:
        balancer.step()
    return float(norm)


def make_balancer(model, enabled=True):
    if not enabled:
        return None
    from train.engine.balancer import NoAuxBalancer
    return NoAuxBalancer(model)


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def truncate_at_eos(tokens: Sequence[int], prompt_len: int, eos_token_id: int) -> Tuple[List[int], bool]:
    """Cut a sampled sequence after the first EOS at or after ``prompt_len``."""
    tokens = list(tokens)
    for index in range(prompt_len, len(tokens)):
        if tokens[index] == eos_token_id:
            return tokens[: index + 1], True
    return tokens, False


def pad_batch(sequences: Sequence[Sequence[int]], pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    width = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(list(seq), dtype=torch.long)
        attention[row, : len(seq)] = True
    return ids.to(device), attention.to(device)


def action_mask_for(prompt_lens: Sequence[int], lengths: Sequence[int], width: int, device) -> torch.Tensor:
    """Float mask over log-prob positions ``t`` (predicting token ``t+1``) of response tokens."""
    mask = torch.zeros((len(lengths), max(width - 1, 0)), dtype=torch.float32)
    for row, (prompt_len, length) in enumerate(zip(prompt_lens, lengths)):
        if prompt_len < 1:
            raise ValueError("a rollout prompt needs at least one token")
        mask[row, prompt_len - 1: length - 1] = 1.0
    return mask.to(device)


def whiten(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to zero mean and unit variance."""
    var = x.var(unbiased=False)
    return (x - x.mean()) / (torch.sqrt(var).clamp_min(1e-6))


def whiten_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) / torch.sqrt(var).clamp_min(1e-6) * mask


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-level Generalized Advantage Estimation.

    ``rewards``, ``values`` and ``mask`` are ``[B, T]``; each row's response is
    one contiguous run of ``mask == 1``. The value after the last response
    token is 0 (terminal). Returns ``(advantages, returns)``, zero off-mask.
    """
    mask = mask.to(values.dtype)
    batch, steps = rewards.shape
    advantages = torch.zeros_like(values)
    running = torch.zeros(batch, dtype=values.dtype, device=values.device)
    for t in reversed(range(steps)):
        if t + 1 < steps:
            next_mask = mask[:, t + 1]
            next_value = values[:, t + 1] * next_mask
        else:
            next_mask = torch.zeros_like(running)
            next_value = torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = delta + gamma * lam * next_mask * running
        advantages[:, t] = running * mask[:, t]
    returns = (advantages + values) * mask
    return advantages, returns


def ppo_step_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Token-level clipped surrogate + per-token k3 KL penalty, masked mean."""
    surrogate, ratio, clip_hit = token_clipped_surrogate(new_logp, old_logp, advantages, clip_eps)
    kl = k3_kl(new_logp, ref_logp)
    total = masked_mean(surrogate + kl_beta * kl, mask)
    return total, {
        "policy_loss": masked_mean(surrogate, mask).detach(),
        "kl": masked_mean(kl, mask).detach(),
        "clip_frac": masked_mean(clip_hit, mask).detach(),
        "approx_kl_old": masked_mean(old_logp - new_logp, mask).detach(),
    }


def _chunks(count: int, size: Optional[int]):
    size = count if not size or size < 1 else size
    for start in range(0, count, size):
        yield list(range(start, min(count, start + size)))


@torch.no_grad()
def _sample(policy, prompt: torch.Tensor, max_new_tokens, temperature, top_p, eos_token_id, use_amp):
    policy.eval()
    with amp_autocast(prompt.device, use_amp):
        return policy.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                               top_p=top_p, eos_token_id=eos_token_id)


@torch.no_grad()
def _frozen_token_logprobs(model, ids: torch.Tensor, use_amp: bool, target_device) -> torch.Tensor:
    device = model_device(model)
    with amp_autocast(device, use_amp):
        return token_logprobs(model, ids.to(device)).float().to(target_device)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    prompt_lens: List[int]
    lengths: List[int]
    full_ids: torch.Tensor          # [B, L] (pad after each sequence)
    attention_mask: torch.Tensor    # [B, L] true on real tokens
    action_mask: torch.Tensor       # [B, L-1] float, response tokens
    old_logp: torch.Tensor          # [B, L-1] policy log-probs at rollout time
    ref_logp: torch.Tensor          # [B, L-1]
    old_values: torch.Tensor        # [B, L-1] value predictions at rollout time
    rewards: torch.Tensor           # [B] reward-model scores
    advantages: torch.Tensor        # [B, L-1] whitened GAE
    returns: torch.Tensor           # [B, L-1]
    finished: torch.Tensor          # [B] bool, EOS produced

    def sub(self, rows: List[int]) -> "RolloutBatch":
        index = torch.tensor(rows, device=self.full_ids.device)
        width = max(self.lengths[r] for r in rows)
        pick = lambda t, w: t.index_select(0, index.to(t.device))[:, :w]
        return RolloutBatch(
            prompt_lens=[self.prompt_lens[r] for r in rows],
            lengths=[self.lengths[r] for r in rows],
            full_ids=pick(self.full_ids, width),
            attention_mask=pick(self.attention_mask, width),
            action_mask=pick(self.action_mask, width - 1),
            old_logp=pick(self.old_logp, width - 1),
            ref_logp=pick(self.ref_logp, width - 1),
            old_values=pick(self.old_values, width - 1),
            rewards=self.rewards.index_select(0, index.to(self.rewards.device)),
            advantages=pick(self.advantages, width - 1),
            returns=pick(self.returns, width - 1),
            finished=self.finished.index_select(0, index.to(self.finished.device)),
        )


class PPOTrainer:
    """Coordinates policy, reference, reward, and value networks during PPO training."""

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        value_model: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        clip_eps: float = 0.2,
        value_clip: float = 0.2,
        kl_beta: float = 0.02,
        value_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        gamma: float = 1.0,
        lam: float = 0.95,
        ppo_epochs: int = 2,
        micro_batch_size: int = 1,
        use_amp: bool = True,
        pad_token_id: int = 0,
        balance: bool = True,
    ):
        if ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1")
        self.policy = policy_model
        self.ref = ref_model.eval()
        self.reward_model = reward_model.eval()
        self.value_model = value_model
        self.policy_opt = policy_optimizer
        self.value_opt = value_optimizer
        self.clip_eps = clip_eps
        self.value_clip = value_clip
        self.kl_beta = kl_beta
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.micro_batch_size = micro_batch_size
        self.use_amp = use_amp
        self.pad_token_id = pad_token_id
        self.policy_scaler = make_grad_scaler(model_device(policy_model), use_amp)
        self.value_scaler = make_grad_scaler(model_device(value_model), use_amp)
        self.policy_balancer = make_balancer(policy_model, balance)
        self.value_balancer = make_balancer(value_model, balance)
        for p in self.ref.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def generate_rollout(
        self,
        prompts: List[List[int]],
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> RolloutBatch:
        if eos_token_id is None:
            raise ValueError("PPO rollouts need eos_token_id to terminate responses")
        device = model_device(self.policy)
        sequences, prompt_lens, finished = [], [], []
        for prompt in prompts:
            prompt = list(prompt)
            out = _sample(self.policy, torch.tensor([prompt], dtype=torch.long, device=device),
                          max_new_tokens, temperature, top_p, eos_token_id, self.use_amp)
            seq, done = truncate_at_eos(out[0].tolist(), len(prompt), eos_token_id)
            sequences.append(seq)
            prompt_lens.append(len(prompt))
            finished.append(done)
        lengths = [len(s) for s in sequences]
        ids, attention = pad_batch(sequences, self.pad_token_id, device)
        mask = action_mask_for(prompt_lens, lengths, ids.size(1), device)

        width = ids.size(1)
        old_logp = torch.zeros((len(sequences), width - 1), device=device)
        ref_logp = torch.zeros_like(old_logp)
        values = torch.zeros_like(old_logp)
        rewards = torch.zeros(len(sequences), device=device)
        self.policy.eval()
        self.value_model.eval()
        vdev, rdev = model_device(self.value_model), model_device(self.reward_model)
        for rows in _chunks(len(sequences), self.micro_batch_size):
            w = max(lengths[r] for r in rows)
            sub_ids, sub_att = ids[rows, :w], attention[rows, :w]
            old_logp[rows, : w - 1] = _frozen_token_logprobs(self.policy, sub_ids, self.use_amp, device)
            ref_logp[rows, : w - 1] = _frozen_token_logprobs(self.ref, sub_ids, self.use_amp, device)
            with amp_autocast(vdev, self.use_amp):
                v = self.value_model(sub_ids.to(vdev), per_token=True)[:, :-1]
            values[rows, : w - 1] = v.float().to(device)
            with amp_autocast(rdev, self.use_amp):
                r = self.reward_model(sub_ids.to(rdev), attention_mask=sub_att.to(rdev))
            rewards[rows] = r.float().to(device)

        token_rewards = torch.zeros_like(values)
        for row, length in enumerate(lengths):
            token_rewards[row, length - 2] = rewards[row]
        values = values * mask
        advantages, returns = compute_gae(token_rewards, values, mask, self.gamma, self.lam)
        advantages = whiten_masked(advantages, mask)
        return RolloutBatch(
            prompt_lens=prompt_lens, lengths=lengths, full_ids=ids, attention_mask=attention,
            action_mask=mask, old_logp=old_logp * mask, ref_logp=ref_logp * mask, old_values=values,
            rewards=rewards, advantages=advantages, returns=returns,
            finished=torch.tensor(finished, device=device),
        )

    def train_step(self, rollout: RolloutBatch) -> Dict[str, float]:
        """``ppo_epochs`` optimizer updates over one rollout (one per epoch, micro-batched)."""
        total_tokens = rollout.action_mask.sum().clamp_min(1.0)
        pdev, vdev = model_device(self.policy), model_device(self.value_model)
        history = []
        for _epoch in range(self.ppo_epochs):
            self.policy.train()
            self.value_model.train()
            sums = {"policy_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "approx_kl_old": 0.0, "value_loss": 0.0}
            for rows in _chunks(len(rollout.lengths), self.micro_batch_size):
                mb = rollout.sub(rows)
                weight = mb.action_mask.sum() / total_tokens
                with amp_autocast(pdev, self.use_amp):
                    new_logp = token_logprobs(self.policy, mb.full_ids)
                loss_p, parts = ppo_step_loss(new_logp.float(), mb.old_logp, mb.ref_logp, mb.advantages,
                                              mb.action_mask, self.clip_eps, self.kl_beta)
                self.policy_scaler.scale(loss_p * weight).backward()

                with amp_autocast(vdev, self.use_amp):
                    v = self.value_model(mb.full_ids.to(vdev), per_token=True)[:, :-1].float()
                old_v, ret, m = mb.old_values.to(vdev), mb.returns.to(vdev), mb.action_mask.to(vdev)
                v_clipped = old_v + (v - old_v).clamp(-self.value_clip, self.value_clip)
                value_loss = masked_mean(0.5 * torch.max((v - ret) ** 2, (v_clipped - ret) ** 2), m)
                self.value_scaler.scale(self.value_coef * value_loss * weight.to(vdev)).backward()

                w = float(weight)
                for key, value in parts.items():
                    sums[key] += float(value) * w
                sums["value_loss"] += float(value_loss.detach()) * w
            optimizer_update(self.policy, self.policy_opt, self.policy_scaler,
                             self.max_grad_norm, self.policy_balancer)
            optimizer_update(self.value_model, self.value_opt, self.value_scaler,
                             self.max_grad_norm, self.value_balancer)
            history.append(sums)
        metrics = dict(history[-1])
        metrics["approx_kl_old_first_epoch"] = history[0]["approx_kl_old"]
        metrics["reward_mean"] = float(rollout.rewards.mean())
        metrics["eos_frac"] = float(rollout.finished.float().mean())
        metrics["response_len"] = float(rollout.action_mask.sum(-1).mean())
        return metrics


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Normalize rewards inside each prompt's sample group. No value network."""
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    scale = grouped.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (centered / scale).view(-1)


@dataclass
class GRPOBatch:
    prompt_lens: List[int]
    lengths: List[int]
    full_ids: torch.Tensor      # [B, L]
    action_mask: torch.Tensor   # [B, L-1]
    old_logp: torch.Tensor      # [B, L-1]
    ref_logp: torch.Tensor      # [B, L-1]
    rewards: torch.Tensor       # [B]
    advantages: torch.Tensor    # [B]
    finished: torch.Tensor      # [B]
    texts: List[str]


class GRPOTrainer:
    """Group Relative Policy Optimization: one policy, one frozen reference, rule rewards."""

    def __init__(self, policy, ref, optimizer, group_size=4, clip_eps=0.2, kl_beta=0.04,
                 epochs=1, micro_batch_size=None, use_amp=True, eos_token_id=None,
                 pad_token_id=None, max_grad_norm=1.0, balance=True):
        if group_size < 2 or epochs < 1:
            raise ValueError("GRPO needs group_size >= 2 and epochs >= 1")
        self.policy = policy
        self.ref = ref.eval()
        self.optimizer = optimizer
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.epochs = epochs
        self.micro_batch_size = micro_batch_size
        self.use_amp = use_amp
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_grad_norm = max_grad_norm
        self.scaler = make_grad_scaler(model_device(policy), use_amp)
        self.balancer = make_balancer(policy, balance)
        for param in self.ref.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def rollout(self, prompts, golds, tokenizer, max_new_tokens=64, temperature=1.0, top_p=1.0) -> GRPOBatch:
        from train.rule_reward import rule_reward
        eos = self.eos_token_id if self.eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("GRPO needs an EOS id to stop completions")
        pad = self.pad_token_id if self.pad_token_id is not None else eos
        device = model_device(self.policy)
        sequences, prompt_lens, texts, finished, rewards = [], [], [], [], []
        for prompt, gold in zip(prompts, golds):
            prompt = prompt.view(-1).tolist() if torch.is_tensor(prompt) else list(prompt)
            batch = torch.tensor([prompt], dtype=torch.long, device=device).repeat(self.group_size, 1)
            out = _sample(self.policy, batch, max_new_tokens, temperature, top_p, eos, self.use_amp)
            for row in out.tolist():
                seq, done = truncate_at_eos(row, len(prompt), eos)
                response = seq[len(prompt):]
                if done:
                    response = response[:-1]
                text = tokenizer.decode(response)
                sequences.append(seq)
                prompt_lens.append(len(prompt))
                finished.append(done)
                texts.append(text)
                rewards.append(rule_reward(text, gold))
        lengths = [len(s) for s in sequences]
        ids, _attention = pad_batch(sequences, pad, device)
        mask = action_mask_for(prompt_lens, lengths, ids.size(1), device)
        old_logp = torch.zeros(mask.shape, device=device)
        ref_logp = torch.zeros_like(old_logp)
        for rows in _chunks(len(sequences), self.micro_batch_size):
            w = max(lengths[r] for r in rows)
            old_logp[rows, : w - 1] = _frozen_token_logprobs(self.policy, ids[rows, :w], self.use_amp, device)
            ref_logp[rows, : w - 1] = _frozen_token_logprobs(self.ref, ids[rows, :w], self.use_amp, device)
        reward_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        return GRPOBatch(
            prompt_lens=prompt_lens, lengths=lengths, full_ids=ids, action_mask=mask,
            old_logp=old_logp * mask, ref_logp=ref_logp * mask, rewards=reward_t,
            advantages=group_advantages(reward_t, self.group_size),
            finished=torch.tensor(finished, device=device), texts=texts,
        )

    def update(self, batch: GRPOBatch) -> Dict[str, float]:
        count = len(batch.lengths)
        device = model_device(self.policy)
        history = []
        for _epoch in range(self.epochs):
            self.policy.train()
            sums = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
            for rows in _chunks(count, self.micro_batch_size):
                w = max(batch.lengths[r] for r in rows)
                ids = batch.full_ids[rows, :w]
                mask = batch.action_mask[rows, : w - 1]
                old = batch.old_logp[rows, : w - 1]
                ref = batch.ref_logp[rows, : w - 1]
                adv = batch.advantages[rows].unsqueeze(-1)
                with amp_autocast(device, self.use_amp):
                    new = token_logprobs(self.policy, ids).float()
                surrogate, _ratio, clip_hit = token_clipped_surrogate(new, old, adv, self.clip_eps)
                kl = k3_kl(new, ref)
                per_seq = masked_mean(surrogate + self.kl_beta * kl, mask, dim=-1)
                loss = per_seq.sum() / count
                self.scaler.scale(loss).backward()
                share = len(rows) / count
                sums["loss"] += float(loss.detach())
                sums["policy_loss"] += float(masked_mean(surrogate, mask, dim=-1).mean().detach()) * share
                sums["kl"] += float(masked_mean(kl, mask, dim=-1).mean().detach()) * share
                sums["clip_frac"] += float(masked_mean(clip_hit, mask, dim=-1).mean()) * share
            optimizer_update(self.policy, self.optimizer, self.scaler, self.max_grad_norm, self.balancer)
            history.append(sums)
        metrics = dict(history[-1])
        metrics["reward_mean"] = float(batch.rewards.mean())
        metrics["eos_frac"] = float(batch.finished.float().mean())
        metrics["response_len"] = float(batch.action_mask.sum(-1).mean())
        return metrics

    def step(self, prompts, golds, tokenizer, max_new_tokens=64, temperature=1.0, top_p=1.0):
        batch = self.rollout(prompts, golds, tokenizer, max_new_tokens, temperature, top_p)
        return self.update(batch)


__all__ = [
    "PPOTrainer", "GRPOTrainer", "RolloutBatch", "GRPOBatch", "compute_gae", "group_advantages",
    "ppo_step_loss", "truncate_at_eos", "pad_batch", "action_mask_for", "whiten", "whiten_masked",
    "amp_autocast", "make_grad_scaler", "prepare_frozen", "optimizer_update", "make_balancer",
    "model_device", "clipped_policy_loss",
]
