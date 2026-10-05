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


def sft_prompt(ids, labels):
    """Prompt of an encoded SFT row: every token before the first supervised label.

    ``tokenize_v2.encode_messages`` masks headers and non-assistant bodies with
    -100, so this is the conversation up to and including the first
    ``<|assistant|>\\n`` header. The assistant answer is never included.
    Returns ``None`` when the row has no supervised token.
    """
    if len(ids) != len(labels):
        raise ValueError("input_ids and labels differ in length")
    for index, label in enumerate(labels):
        if label != -100:
            return list(ids[:index]) if index > 0 else None
    return None


def fit_preference(chosen, rejected, prompt_len, limit):
    """Fit a pair while preserving one identical prompt boundary.

    Trimming chosen and rejected independently changes the prompt tokens and
    makes DPO compare different conditioning contexts.  Keep the shared prompt
    tail and the same-length response prefix for both records.
    """
    chosen = list(chosen)
    rejected = list(rejected)
    if limit < 2:
        raise ValueError("preference limit must leave a prompt and one answer token")
    if prompt_len <= 0:
        prompt_len = 0
        for left, right in zip(chosen, rejected):
            if left != right:
                break
            prompt_len += 1
    prompt_len = min(prompt_len, len(chosen), len(rejected))
    if prompt_len == 0 or chosen[:prompt_len] != rejected[:prompt_len]:
        return None
    prompt_keep = min(prompt_len, limit - 1)
    prompt = chosen[prompt_len - prompt_keep:prompt_len]
    response_limit = limit - prompt_keep
    chosen_response = chosen[prompt_len:prompt_len + response_limit]
    rejected_response = rejected[prompt_len:prompt_len + response_limit]
    if not chosen_response or not rejected_response:
        return None
    return prompt + chosen_response, prompt + rejected_response, prompt_keep
