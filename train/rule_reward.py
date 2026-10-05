"""Format and answer checks for GRPO. No model import.

Reward = 0.5 for a ``<think> ... </think>`` block, +0.5 for a non-empty answer
after it, +1.0 when the answer equals the gold string after normalisation.
The answer candidate is the whole text after ``</think>`` or, when present,
the content of the last ``\\boxed{...}`` in it (OpenR1 solutions end with a
boxed result). Substrings never count as a match.
"""

import re


def rule_reward(text, gold=None):
    if not isinstance(text, str):
        return 0.0
    start = text.find("<think>")
    end = text.find("</think>")
    if start < 0 or end < start:
        return 0.0
    reward = 0.5
    answer = text[end + len("</think>"):].strip()
    if answer:
        reward += 0.5
    if gold is not None and answer:
        target = _normalise(str(gold))
        candidates = [answer]
        boxed = last_boxed(answer)
        if boxed is not None:
            candidates.append(boxed)
        if target and any(_normalise(c) == target for c in candidates):
            reward += 1.0
    return reward


def last_boxed(text):
    """Content of the last ``\\boxed{...}`` with balanced braces, else ``None``."""
    index = text.rfind("\\boxed{")
    if index < 0:
        return None
    position = index + len("\\boxed{")
    depth = 1
    for cursor in range(position, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[position:cursor]
    return None


def _normalise(value: str) -> str:
    value = value.strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value.strip("$")
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.strip(" .。!！?？")
