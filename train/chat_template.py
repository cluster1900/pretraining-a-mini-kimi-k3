"""Chat packing that reproduces ``train/data/tokenize_v2.encode_messages`` exactly.

Ground truth (audited, hash-bound, read-only): for every turn

    header = encode("<|{role}|>\\n", append_eos=False)          # label -100
    body   = encode(content + "\\n", append_eos=role == "assistant")
    labels = body for assistant turns, -100 otherwise

Header and body are encoded separately, so BPE never merges across the
header/body boundary. ``pack_chat`` uses the same calls so a conversation packed
here is token-identical to an SFT training row. ``add_generation_prompt``
appends the assistant header for inference.

Note: ``tokenize_v2.encode_preference`` (UltraFeedback rows) encodes prompt turns
as ONE string ``"<|role|>\\n" + content + "\\n"`` and responses without the
trailing newline. That differs from the SFT template; inference follows the
SFT template.

Truncation keeps the tail: whole turns are dropped from the front (oldest
first). A leading system turn is kept when it still fits. The newest user turn
(and everything after it) is never dropped; if it cannot fit, ``ValueError``.
"""
from typing import Dict, List, Tuple


def encode_turn(turn: Dict[str, str], tokenizer) -> Tuple[List[int], List[int]]:
    role = turn["role"]
    header = tokenizer.encode(f"<|{role}|>\n", append_eos=False)
    body = tokenizer.encode(turn["content"] + "\n", append_eos=role == "assistant")
    labels = [-100] * len(header) + (list(body) if role == "assistant" else [-100] * len(body))
    return list(header) + list(body), labels


def assistant_header(tokenizer) -> List[int]:
    return list(tokenizer.encode("<|assistant|>\n", append_eos=False))


def pack_chat(messages: List[Dict[str, str]], tokenizer, max_length: int,
              add_generation_prompt: bool = False) -> Tuple[list, list]:
    """Return ``(input_ids, labels)``; only assistant bodies and their EOS are supervised."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    turns = [encode_turn(m, tokenizer) for m in messages]
    tail_ids = assistant_header(tokenizer) if add_generation_prompt else []
    budget = max_length - len(tail_ids)

    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    must_keep_from = last_user if last_user is not None else max(len(turns) - 1, 0)
    required = sum(len(ids) for ids, _ in turns[must_keep_from:])
    if required > budget:
        raise ValueError(
            f"the newest user turn needs {required + len(tail_ids)} tokens, above max_length={max_length}"
        )
    start = must_keep_from
    used = required
    keep_system = (start > 0 and messages[0].get("role") == "system"
                   and used + len(turns[0][0]) <= budget)
    floor = 1 if keep_system else 0
    if keep_system:
        used += len(turns[0][0])
    while start > floor and used + len(turns[start - 1][0]) <= budget:
        start -= 1
        used += len(turns[start][0])
    keep = ([0] if keep_system else []) + list(range(start, len(turns)))

    ids, labels = [], []
    for index in keep:
        ids.extend(turns[index][0])
        labels.extend(turns[index][1])
    ids.extend(tail_ids)
    labels.extend([-100] * len(tail_ids))
    return ids, labels
