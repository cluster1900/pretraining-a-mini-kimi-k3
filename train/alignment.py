"""SFT, preference (DPO) and lightweight PPO utilities for Mini-K3.

These utilities share the causal model and keep alignment training inside
``train/``.  They are intentionally model-agnostic: a checkpoint from the
pretraining script can be loaded directly for SFT or preference tuning.
"""
import torch
import torch.nn.functional as F


def causal_sft_loss(logits, labels, loss_mask=None):
    """Next-token CE for packed conversations; mask prompt tokens with 0."""
    l = logits[..., :-1, :].contiguous()
    y = labels[..., 1:].contiguous()
    per_token = F.cross_entropy(l.view(-1, l.size(-1)), y.view(-1), reduction="none")
    y_flat = y.view(-1)
    if loss_mask is None:
        loss_mask = (y_flat != -100).float()
    else:
        loss_mask = loss_mask[..., 1:].contiguous().float().view(-1)
    valid = loss_mask * (y_flat != -100).float()
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0)


def sequence_logprob(logits, labels, attention_mask=None, average_logprobs=False):
    """Log p(y|x) for sequence, used by DPO and reward-model checks."""
    lp = F.log_softmax(logits[..., :-1, :].float(), dim=-1)
    y = labels[..., 1:]
    # Clamp negative ignore indices (e.g. -100) to 0 to avoid negative indexing or NaN propagation
    y_safe = y.clamp_min(0)
    tok = lp.gather(-1, y_safe.unsqueeze(-1)).squeeze(-1)
    mask = (y != -100).float()
    if attention_mask is not None:
        mask = mask * attention_mask[..., 1:].float()
    seq_logp = (tok * mask).sum(-1)
    if average_logprobs:
        return seq_logp / mask.sum(-1).clamp_min(1.0)
    return seq_logp


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Direct Preference Optimization objective for chosen/rejected pairs."""
    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def clipped_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """PPO clipped surrogate; rollout/reward generation remains external."""
    ratio = torch.exp(new_logp - old_logp)
    return -torch.min(ratio * advantages,
                      ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages).mean()


def pairwise_reward_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry preference objective for Reward Model training:
    loss = -log sigmoid(r_chosen - r_rejected)
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

