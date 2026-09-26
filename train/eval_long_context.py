"""
Long-Context Evaluation: Needle In A Haystack (NIAH) & Passkey Retrieval.
Enforces Rule 8 of AGENTS.md:
"1M 推理能力必须同时通过无 cache/cache logits 等价测试、长文档评测和显存基准；仅提高位置上限或创建 cache 数据结构不视为完成。"

Evaluates the model's ability to recall a specific needle (passkey) inserted
at varying depths (10%, 25%, 50%, 75%, 90%) across long contexts (4K to 1M).
"""

import sys
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.data.tokenizer import K3Tokenizer


FILLER_TEXTS = [
    "The atmospheric pressure at sea level is approximately 1013.25 hectopascals under standard conditions. ",
    "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs. ",
    "Silicon has atomic number 14 and is a tetravalent metalloid and semiconductor used widely in electronics. ",
    "Photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy. ",
    "The speed of light in a vacuum is defined as exactly 299,792,458 meters per second by international convention. ",
]


def generate_haystack(target_tokens: int, tokenizer: K3Tokenizer) -> Tuple[str, List[int]]:
    """Generates synthetic background text up to target token count."""
    tokens = []
    text_pieces = []
    idx = 0
    while len(tokens) < target_tokens:
        snippet = FILLER_TEXTS[idx % len(FILLER_TEXTS)]
        snippet_ids = tokenizer.encode(snippet, append_eos=False)
        tokens.extend(snippet_ids)
        text_pieces.append(snippet)
        idx += 1
    return "".join(text_pieces), tokens[:target_tokens]


def construct_niah_prompt(
    target_len: int,
    depth_frac: float,
    passkey: str,
    tokenizer: K3Tokenizer,
) -> Tuple[List[int], str]:
    """
    Inserts needle: 'The secret passkey is {passkey}. Remember this number.'
    at specified depth fraction of the haystack.
    """
    prefix = "Below is a long document containing technical facts. Read it carefully.\n\n"
    prefix_ids = tokenizer.encode(prefix, append_eos=False)

    needle = f"\n[SPECIAL MEMORANDUM] The secret passkey is {passkey}. Remember this passkey verbatim.\n"
    needle_ids = tokenizer.encode(needle, append_eos=False)

    suffix = "\n\nBased on the text above, what is the secret passkey? The secret passkey is "
    suffix_ids = tokenizer.encode(suffix, append_eos=False)

    haystack_len = max(0, target_len - len(prefix_ids) - len(needle_ids) - len(suffix_ids))
    _, haystack_tokens = generate_haystack(haystack_len, tokenizer)

    # Split haystack according to depth fraction
    split_point = int(len(haystack_tokens) * depth_frac)
    part1 = haystack_tokens[:split_point]
    part2 = haystack_tokens[split_point:]

    full_ids = prefix_ids + part1 + needle_ids + part2 + suffix_ids
    return full_ids, passkey


def evaluate_needle_retrieval(
    model: MiniK3ForCausalLM,
    tokenizer: K3Tokenizer,
    target_lengths: List[int],
    depths: List[float],
    device: torch.device,
    max_new_tokens: int = 8,
) -> Dict[Tuple[int, float], bool]:
    """
    Runs NIAH evaluation across given lengths and depths using sliding window and KDA recurrence.
    """
    model.eval()
    results = {}

    print("\n" + "=" * 80)
    print(" Running Needle-In-A-Haystack (NIAH) Long-Context Benchmark")
    print("=" * 80)
    print(f"{'Length':>10} | {'Depth':>8} | {'Passkey':>10} | {'Output':>16} | {'Status':>10}")
    print("-" * 80)

    for l in target_lengths:
        for d in depths:
            passkey = str(random.randint(10000, 99999))
            prompt_ids, expected = construct_niah_prompt(l, d, passkey, tokenizer)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Generate response
            with torch.no_grad():
                gen_ids = model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,  # Greedy deterministic
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_text = tokenizer.decode(gen_ids[0, len(prompt_ids):].tolist()).strip()
            passed = expected in gen_text
            results[(l, d)] = passed
            status_str = "PASS" if passed else "FAIL"
            print(f"{l:10,d} | {d * 100:7.1f}% | {expected:>10} | {gen_text[:16]:>16} | {status_str:>10}")

    print("-" * 80)
    total_runs = len(results)
    passed_runs = sum(1 for v in results.values() if v)
    print(f"[*] Retrieval Accuracy: {passed_runs}/{total_runs} ({passed_runs / max(1, total_runs) * 100:.1f}%)")
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(description="Mini K3 Needle In A Haystack Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or model.pt")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384], help="Context lengths to evaluate")
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Needle depths (fractions from 0.0 to 1.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = DEFAULT_CONFIG
    tokenizer = K3Tokenizer()

    print("=" * 80)
    print(f" Mini Kimi K3: Long-Context Retrieval Benchmark (AGENTS.md Rule 8)")
    print("=" * 80)

    model = MiniK3ForCausalLM(cfg).to(device)
    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        model_file = ckpt_p / "model.pt" if ckpt_p.is_dir() else ckpt_p
        if not model_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {model_file}")
        print(f"[*] Loading weights from: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
    else:
        print("[*] No checkpoint specified. Running architectural pipeline test on initialized model.")

    evaluate_needle_retrieval(model, tokenizer, args.lengths, args.depths, device)


if __name__ == "__main__":
    main()
