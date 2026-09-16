"""Simple deterministic chat packing for SFT and rollout datasets."""
from typing import List, Dict, Tuple

def pack_chat(messages: List[Dict[str, str]], tokenizer, max_length: int) -> Tuple[list, list]:
    """Return input ids and labels; only assistant content and turn EOS contribute to loss."""
    ids, labels = [], []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            # Mask the assistant role header; train only response tokens and terminating EOS
            header = f"<|{role}|>\n"
            header_ids = tokenizer.encode(header, append_eos=False)
            body_ids = tokenizer.encode(f"{content}\n", append_eos=True)
            ids.extend(header_ids + body_ids)
            labels.extend([-100] * len(header_ids) + body_ids)
        else:
            # User and system turns do NOT terminate with EOS
            text = f"<|{role}|>\n{content}\n"
            part = tokenizer.encode(text, append_eos=False)
            ids.extend(part)
            labels.extend([-100] * len(part))
    ids, labels = ids[:max_length], labels[:max_length]
    return ids, labels
