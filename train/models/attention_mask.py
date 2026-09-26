"""Causal sliding-window mask shared by MLA attention and its tests."""


def attention_blocked(q_pos, k_pos, window: int):
    """True where a query must not attend: future tokens, or tokens outside the window."""
    return (k_pos > q_pos) | (k_pos + window <= q_pos)


def window_key_start(query_start: int, window: int) -> int:
    """First key index visible to the first query in a block."""
    return max(0, query_start - window + 1)
