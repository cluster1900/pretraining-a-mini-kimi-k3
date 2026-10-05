"""SFT, preference (DPO) and policy-gradient utilities for Mini-K3.

These utilities share the causal model and keep alignment training inside
``train/``.  They are intentionally model-agnostic: a checkpoint from the
pretraining script can be loaded directly for SFT or preference tuning.

Token log-probabilities are computed from the final hidden states in chunks
(``token_logprobs``) so the full ``[tokens, 163840]`` FP32 logits are never
resident; under autograd each chunk is recomputed during backward.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


def _chunk_token_logprob(head, hidden, targets):
    logits = head(hidden).float()
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


def token_logprobs(model, input_ids, chunk_size=1024):
    """``out[:, t] = log p(input_ids[:, t+1] | input_ids[:, :t+1])`` as FP32 ``[B, L-1]``.

    Uses ``model(..., return_hidden=True, compute_logits=False)`` and applies
    ``model.lm_head`` chunk by chunk (recomputed in backward when grad is on).
    """
    hidden = model(input_ids, return_hidden=True, compute_logits=False)["hidden"]
    batch, length, width = hidden.shape
    if length < 2:
        return hidden.new_zeros((batch, 0), dtype=torch.float32)
    flat_h = hidden[:, :-1].reshape(-1, width)
    flat_y = input_ids[:, 1:].reshape(-1)
    head = model.lm_head
    pieces = []
    for start in range(0, flat_h.size(0), chunk_size):
        h = flat_h[start:start + chunk_size]
        y = flat_y[start:start + chunk_size]
        if torch.is_grad_enabled() and h.requires_grad:
            pieces.append(checkpoint(_chunk_token_logprob, head, h, y, use_reentrant=False))
        else:
            pieces.append(_chunk_token_logprob(head, h, y))
    return torch.cat(pieces).view(batch, length - 1)


def masked_mean(values, mask, dim=None):
    mask = mask.to(values.dtype)
    if dim is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    return (values * mask).sum(dim) / mask.sum(dim).clamp_min(1.0)


def k3_kl(new_logp, ref_logp, max_log_ratio=20.0):
    """Per-token non-negative KL(new || ref) estimator k3 = exp(r) - r - 1, r = ref - new."""
    r = (ref_logp - new_logp).clamp(-max_log_ratio, max_log_ratio)
    return torch.exp(r) - r - 1.0


def response_logprob(token_lp, response_mask, length_norm=False):
    """Sum (or mean when ``length_norm``) of token log-probs over the response mask."""
    mask = response_mask.to(token_lp.dtype)
    total = (token_lp * mask).sum(-1)
    if length_norm:
        return total / mask.sum(-1).clamp_min(1.0)
    return total


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Direct Preference Optimization objective for chosen/rejected pairs."""
    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def dpo_metrics(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Implicit rewards beta*(log pi - log ref) and their margin / accuracy."""
    chosen = beta * (policy_chosen - ref_chosen).detach()
    rejected = beta * (policy_rejected - ref_rejected).detach()
    return {
        "reward_chosen": float(chosen.mean()),
        "reward_rejected": float(rejected.mean()),
        "reward_margin": float((chosen - rejected).mean()),
        "reward_acc": float((chosen > rejected).float().mean()),
    }


def clipped_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """PPO clipped surrogate on already-aggregated log-probs (kept for callers/tests)."""
    ratio = torch.exp(new_logp - old_logp)
    return -torch.min(ratio * advantages,
                      ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages).mean()


def token_clipped_surrogate(new_logp, old_logp, advantages, clip_eps=0.2):
    """Per-token PPO clipped surrogate loss (not reduced). Shapes broadcast."""
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(unclipped, clipped)
    clip_hit = ((ratio - 1.0).abs() > clip_eps).float()
    return loss, ratio, clip_hit


def pairwise_reward_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry preference objective for Reward Model training:
    loss = -log sigmoid(r_chosen - r_rejected)
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
