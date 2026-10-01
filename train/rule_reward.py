"""Format and answer checks for GRPO. No model import."""

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
    if gold is not None and _normalise(answer) == _normalise(str(gold)):
        reward += 1.0
    return reward


def _normalise(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.strip(" .。!！?？")
