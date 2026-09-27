"""Trim alignment records to the training sequence length without importing the model."""


def fit_tail(ids, limit):
    """Keep the tail of a sequence. Returns the kept ids and how many tokens were dropped."""
    if limit < 1:
        raise ValueError("sequence limit must be positive")
    if len(ids) <= limit:
        return list(ids), 0
    dropped = len(ids) - limit
    return list(ids[-limit:]), dropped


def fit_sft(ids, labels, limit):
    kept, dropped = fit_tail(ids, limit)
    kept_labels = list(labels[-len(kept):])
    if not any(label != -100 for label in kept_labels[1:]):
        return None
    return kept, kept_labels, dropped


def fit_preference(chosen, rejected, prompt_len, limit):
    chosen_kept, chosen_drop = fit_tail(chosen, limit)
    rejected_kept, rejected_drop = fit_tail(rejected, limit)
    chosen_prompt = max(0, prompt_len - chosen_drop)
    rejected_prompt = max(0, prompt_len - rejected_drop)
    if chosen_prompt >= len(chosen_kept) or rejected_prompt >= len(rejected_kept):
        return None
    return chosen_kept, rejected_kept, chosen_prompt, rejected_prompt
