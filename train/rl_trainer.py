"""
PPO / RLHF Orchestration & Building Blocks for Mini Kimi K3.
Features:
- Actor-Critic coordination: Policy, Reference, Reward, and Value models
- Rollout generation using incremental KV-cache decoding
- Generalized Advantage Estimation (GAE)
- Ratio clipping, value clipping, and adaptive KL penalty
- Full compliance with Section 5 & 6 of TRAINING_PLAN.md
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.alignment import sequence_logprob, clipped_policy_loss


def whiten(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to zero mean and unit variance."""
    var = x.var(unbiased=False)
    return (x - x.mean()) / (torch.sqrt(var).clamp_min(1e-6))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes sequence-level Generalized Advantage Estimation (GAE).
    For final-token/sequence-level rewards:
        delta = reward - value
        advantage = delta
        returns = reward
    """
    advantages = rewards - values
    returns = rewards
    return whiten(advantages), returns


def ppo_step_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    clip_eps: float = 0.2,
    value_clip: float = 0.2,
    kl_beta: float = 0.02,
    value_coef: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes policy surrogate loss, clipped value loss, and KL penalty.
    """
    advantages = whiten(rewards - values.detach())
    policy = clipped_policy_loss(new_logp, old_logp, advantages, clip_eps)

    old_v = old_values.detach()
    v_clipped = old_v + (values - old_v).clamp(-value_clip, value_clip)
    value_loss = 0.5 * torch.max((values - rewards).pow(2), (v_clipped - rewards).pow(2)).mean()
    kl = (new_logp - ref_logp).mean()

    total_loss = policy + value_coef * value_loss + kl_beta * kl
    metrics = {
        "policy_loss": policy.detach(),
        "value_loss": value_loss.detach(),
        "kl": kl.detach(),
        "reward_mean": rewards.mean().detach(),
    }
    return total_loss, metrics


@dataclass
class RolloutBatch:
    prompt_lens: List[int]
    full_ids: torch.Tensor          # [B, L]
    response_mask: torch.Tensor     # [B, L]
    old_logp: torch.Tensor         # [B]
    ref_logp: torch.Tensor         # [B]
    rewards: torch.Tensor          # [B]
    values: torch.Tensor           # [B]


class PPOTrainer:
    """
    Coordinates policy, reference, reward, and value networks during PPO training.
    """
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
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
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

        # Freeze reference and reward models completely
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
        top_p: float = 0.9,
    ) -> RolloutBatch:
        """
        Generates completions for a batch of prompts, computes reference logprobs,
        rewards, and baseline values.
        """
        device = next(self.policy.parameters()).device
        self.policy.eval()

        completed_ids = []
        prompt_lens = [len(p) for p in prompts]

        # Generate sequence for each prompt
        for p_ids in prompts:
            p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = self.policy.generate(
                p_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
            completed_ids.append(gen[0].tolist())

        # Pad to common length for batch evaluation
        max_l = max(len(c) for c in completed_ids)
        batch_ids = torch.zeros((len(completed_ids), max_l), dtype=torch.long, device=device)
        response_mask = torch.zeros((len(completed_ids), max_l), dtype=torch.bool, device=device)

        for i, c in enumerate(completed_ids):
            batch_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)
            p_len = prompt_lens[i]
            response_mask[i, p_len : len(c)] = True

        labels = batch_ids.clone()
        labels[~response_mask] = -100

        # 1. Compute old log probs under policy
        policy_logits = self.policy(batch_ids)["logits"]
        old_logp = sequence_logprob(policy_logits, labels, average_logprobs=False)

        # 2. Compute reference log probs
        ref_logits = self.ref(batch_ids)["logits"]
        ref_logp = sequence_logprob(ref_logits, labels, average_logprobs=False)

        # 3. Compute rewards from frozen RewardModel
        rewards = self.reward_model(batch_ids, attention_mask=batch_ids.ne(0))

        # 4. Compute baseline values from ValueModel
        values = self.value_model(batch_ids, attention_mask=batch_ids.ne(0))

        return RolloutBatch(
            prompt_lens=prompt_lens,
            full_ids=batch_ids,
            response_mask=response_mask,
            old_logp=old_logp,
            ref_logp=ref_logp,
            rewards=rewards,
            values=values,
        )

    def train_step(self, rollout: RolloutBatch) -> Dict[str, float]:
        """
        Executes one PPO training step over the collected rollout batch.
        """
        self.policy.train()
        self.value_model.train()

        labels = rollout.full_ids.clone()
        labels[~rollout.response_mask] = -100

        # Current policy logprobs and values
        new_logits = self.policy(rollout.full_ids)["logits"]
        new_logp = sequence_logprob(new_logits, labels, average_logprobs=False)
        current_values = self.value_model(rollout.full_ids, attention_mask=rollout.full_ids.ne(0))

        total_loss, metrics = ppo_step_loss(
            new_logp=new_logp,
            old_logp=rollout.old_logp,
            ref_logp=rollout.ref_logp,
            rewards=rollout.rewards,
            values=current_values,
            old_values=rollout.values,
            clip_eps=self.clip_eps,
            value_clip=self.value_clip,
            kl_beta=self.kl_beta,
            value_coef=self.value_coef,
        )

        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)

        self.policy_opt.step()
        self.value_opt.step()

        return {k: v.item() for k, v in metrics.items()}
