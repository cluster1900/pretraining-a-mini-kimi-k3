"""
Long-Context Evaluation: Needle In A Haystack (NIAH) passkey retrieval.
Supports Rule 8 of AGENTS.md:
"1M 推理能力必须同时通过无 cache/cache logits 等价测试、长文档评测和显存基准；仅提高位置上限或创建 cache 数据结构不视为完成。"

A 5-digit passkey is inserted at a depth fraction of a synthetic haystack; the
prompt is prefilled through the KV cache in chunks and the answer is decoded
greedily token by token from the cache.

Defaults (exactly as implemented):
- ``--lengths``: every value of 2048, 4096, 8192, 16384 that does not exceed
  the checkpoint's trained sequence length (read from ``meta.pt`` →
  ``run_signature.sequence_length`` or ``alignment_meta.json``; 2048 when the
  checkpoint carries no metadata).
- ``--depths 0.1 0.5 0.9``, ``--samples 5`` passkeys per (length, depth) cell,
  ``--seed 1234`` (passkeys come from ``random.Random(seed)``),
  ``--prefill_chunk 4096`` tokens per cached prefill call,
  ``--max_new_tokens 8``, FP16 autocast on CUDA (``--no-amp`` disables it).
- Lengths above the trained sequence length are refused unless
  ``--allow-untrained-length`` is passed (results are then labelled
  UNTRAINED and do not count as long-context evidence).
- ``--checkpoint`` may be a run root (newest ``step_*`` with ``COMPLETE`` is
  used; none → error), a step directory, an alignment output directory or a
  ``model.pt`` file.
"""

import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

DEFAULT_LENGTHS = (2048, 4096, 8192, 16384)
UNKNOWN_TRAINED_LENGTH = 2048

FILLER_TEXTS = [
    "The atmospheric pressure at sea level is approximately 1013.25 hectopascals under standard conditions. ",
    "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs. ",
    "Silicon has atomic number 14 and is a tetravalent metalloid and semiconductor used widely in electronics. ",
    "Photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy. ",
    "The speed of light in a vacuum is defined as exactly 299,792,458 meters per second by international convention. ",
]


def generate_haystack(target_tokens: int, tokenizer) -> List[int]:
    """Synthetic background token ids of exactly ``target_tokens`` length."""
    pieces = [tokenizer.encode(text, append_eos=False) for text in FILLER_TEXTS]
    tokens: List[int] = []
    index = 0
    while len(tokens) < target_tokens:
        tokens.extend(pieces[index % len(pieces)])
        index += 1
    return tokens[:target_tokens]


def construct_niah_prompt(target_len: int, depth_frac: float, passkey: str, tokenizer) -> Tuple[List[int], str]:
    """Prompt of ``target_len`` tokens with the passkey needle at ``depth_frac``."""
    prefix_ids = tokenizer.encode("Below is a long document containing technical facts. Read it carefully.\n\n",
                                  append_eos=False)
    needle_ids = tokenizer.encode(
        f"\n[SPECIAL MEMORANDUM] The secret passkey is {passkey}. Remember this passkey verbatim.\n",
        append_eos=False)
    suffix_ids = tokenizer.encode(
        "\n\nBased on the text above, what is the secret passkey? The secret passkey is ", append_eos=False)
    haystack_len = max(0, target_len - len(prefix_ids) - len(needle_ids) - len(suffix_ids))
    haystack = generate_haystack(haystack_len, tokenizer)
    split_point = int(len(haystack) * depth_frac)
    full_ids = prefix_ids + haystack[:split_point] + needle_ids + haystack[split_point:] + suffix_ids
    return full_ids, passkey


def resolve_lengths(requested: Optional[Sequence[int]], trained: Optional[int], allow_untrained: bool) -> List[int]:
    limit = trained if trained is not None else UNKNOWN_TRAINED_LENGTH
    if requested is None:
        lengths = [n for n in DEFAULT_LENGTHS if n <= limit]
        if not lengths:
            lengths = [limit]
        return lengths
    lengths = [int(n) for n in requested]
    too_long = [n for n in lengths if n > limit]
    if too_long and not allow_untrained:
        source = "trained" if trained is not None else "assumed (no checkpoint metadata)"
        raise ValueError(
            f"lengths {too_long} exceed the {source} sequence length {limit}; "
            "pass --allow-untrained-length to evaluate them anyway"
        )
    return lengths


@torch.no_grad()
def prefill_chunked(model, ids: torch.Tensor, chunk: int):
    """Feed ``ids`` through the cache ``chunk`` tokens at a time; return last logits and cache."""
    if chunk < 1:
        raise ValueError("prefill chunk must be positive")
    cache = model.new_kv_cache()
    out = None
    for start in range(0, ids.size(1), chunk):
        out = model(ids[:, start:start + chunk], use_cache=True, past_key_values=cache, logits_to_keep=1)
    return out["logits"][:, -1], cache


@torch.no_grad()
def greedy_continue(model, ids: torch.Tensor, max_new_tokens: int, eos_token_id: Optional[int], chunk: int) -> List[int]:
    logits, cache = prefill_chunked(model, ids, chunk)
    generated: List[int] = []
    for step in range(max_new_tokens):
        next_id = logits.float().argmax(-1, keepdim=True)
        token = int(next_id.item())
        if eos_token_id is not None and token == eos_token_id:
            break
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        logits = model(next_id, use_cache=True, past_key_values=cache)["logits"][:, -1]
    return generated


def evaluate_needle_retrieval(
    model,
    tokenizer,
    target_lengths: List[int],
    depths: List[float],
    device: torch.device,
    max_new_tokens: int = 8,
    samples: int = 5,
    seed: int = 1234,
    prefill_chunk: int = 4096,
    trained_length: Optional[int] = None,
    use_amp: bool = True,
) -> Dict[Tuple[int, float], float]:
    """Return accuracy per (length, depth) cell over ``samples`` seeded passkeys."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    model.eval()
    rng = random.Random(seed)
    results: Dict[Tuple[int, float], float] = {}
    limit = trained_length if trained_length is not None else UNKNOWN_TRAINED_LENGTH
    autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp and device.type == "cuda" else None)

    print("\n" + "=" * 80)
    print(f" Needle-In-A-Haystack | samples/cell={samples} seed={seed} prefill_chunk={prefill_chunk}")
    print("=" * 80)
    print(f"{'Length':>10} | {'Depth':>7} | {'Correct':>9} | {'Accuracy':>8} | Note")
    print("-" * 80)
    total = correct_total = 0
    for length in target_lengths:
        for depth in depths:
            correct = 0
            for _ in range(samples):
                passkey = str(rng.randint(10000, 99999))
                prompt_ids, expected = construct_niah_prompt(length, depth, passkey, tokenizer)
                ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                if autocast is not None:
                    with autocast:
                        out = greedy_continue(model, ids, max_new_tokens, tokenizer.eos_token_id, prefill_chunk)
                else:
                    out = greedy_continue(model, ids, max_new_tokens, tokenizer.eos_token_id, prefill_chunk)
                correct += int(expected in tokenizer.decode(out))
            accuracy = correct / samples
            results[(length, depth)] = accuracy
            total += samples
            correct_total += correct
            note = "UNTRAINED LENGTH" if length > limit else ""
            print(f"{length:10,d} | {depth * 100:6.1f}% | {correct:4d}/{samples:<4d} | {accuracy * 100:7.1f}% | {note}")
    print("-" * 80)
    print(f"[*] Retrieval Accuracy: {correct_total}/{total} ({correct_total / max(1, total) * 100:.1f}%)")
    print("=" * 80)
    return results


def main():
    from train.alignment_models import load_causal_model
    from train.data.tokenizer import K3Tokenizer

    parser = argparse.ArgumentParser(description="Mini K3 Needle In A Haystack Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Run root (newest complete step_*), step dir, alignment dir or model.pt")
    parser.add_argument("--lengths", type=int, nargs="+", default=None,
                        help="Context lengths. Default: 2048/4096/8192/16384 up to the trained length")
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                        help="Needle depths (fractions from 0.0 to 1.0)")
    parser.add_argument("--samples", type=int, default=5, help="Passkeys per (length, depth) cell")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for passkey sampling")
    parser.add_argument("--prefill_chunk", type=int, default=4096, help="Tokens per cached prefill call")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--allow-untrained-length", action="store_true",
                        help="Allow lengths above the checkpoint's trained sequence length")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="FP16 autocast on CUDA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_model", required=True, help="Directory containing the audited tiktoken.model")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = K3Tokenizer(args.tokenizer_model)
    model, info, cfg = load_causal_model(args.checkpoint, device)
    trained = info.sequence_length
    lengths = resolve_lengths(args.lengths, trained, args.allow_untrained_length)
    if max(lengths) > cfg.max_position_embeddings:
        raise ValueError(f"lengths exceed max_position_embeddings={cfg.max_position_embeddings}")

    print("=" * 80)
    print(" Mini Kimi K3: Long-Context Retrieval Benchmark (AGENTS.md Rule 8)")
    print(f" Weights: {info.model_file}")
    print(f" Trained sequence length: {trained if trained is not None else f'unknown (assuming {UNKNOWN_TRAINED_LENGTH})'}")
    print(f" Lengths: {lengths}")
    print("=" * 80)
    evaluate_needle_retrieval(
        model, tokenizer, lengths, args.depths, device, max_new_tokens=args.max_new_tokens,
        samples=args.samples, seed=args.seed, prefill_chunk=args.prefill_chunk,
        trained_length=trained, use_amp=args.amp,
    )


if __name__ == "__main__":
    main()
