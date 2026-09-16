"""
Standalone Causal Multiple-Choice Benchmark Evaluator for Mini Kimi K3.
Features:
- Right-padding (critical for causal models; left-padding corrupts context)
- Length-normalized log-probability scoring: sum(log P(choice_tokens)) / len(choice_tokens)
- Independent read-only evaluation from checkpoint volume
"""

import sys
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F

from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.data.tokenizer import K3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mini K3 Checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (must contain model.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_model", required=True, help="Directory containing tiktoken.model")
    return parser.parse_args()


def score_multiple_choice(
    model: MiniK3ForCausalLM,
    tokenizer: K3Tokenizer,
    prompt: str,
    choices: List[str],
    device: torch.device,
) -> int:
    """
    Evaluates a single multiple-choice question.
    Uses right-padding and length-normalized log-probabilities.
    Returns the predicted choice index (0..len(choices)-1).
    """
    prompt_ids = tokenizer.encode(prompt, append_eos=False)
    prompt_len = len(prompt_ids)

    candidate_ids = []
    choice_lens = []
    for c in choices:
        c_ids = tokenizer.encode(c, append_eos=False)
        full_ids = prompt_ids + c_ids
        candidate_ids.append(full_ids)
        choice_lens.append(len(c_ids))

    # Right-pad to max sequence length in batch
    max_len = max(len(ids) for ids in candidate_ids)
    batch_tensor = torch.zeros((len(choices), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(candidate_ids):
        batch_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out = model(batch_tensor)
        logits = out["logits"]  # [num_choices, max_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

    scores = []
    for i in range(len(choices)):
        c_len = choice_lens[i]
        # Target tokens for this choice
        # Tokens are positioned at prompt_len .. prompt_len + c_len
        # Predictions are at prompt_len - 1 .. prompt_len + c_len - 2
        pred_pos = torch.arange(prompt_len - 1, prompt_len + c_len - 1, device=device)
        target_tokens = torch.tensor(candidate_ids[i][prompt_len:], device=device)

        choice_log_probs = log_probs[i, pred_pos, target_tokens]
        # Length normalization
        score = (choice_log_probs.sum() / max(1, c_len)).item()
        scores.append(score)

    return int(torch.tensor(scores).argmax().item())


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    model_file = ckpt_path / "model.pt" if ckpt_path.is_dir() else ckpt_path

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")

    print("=" * 70)
    print(f" Mini Kimi K3 Benchmark Evaluator")
    print(f" Loading weights from: {model_file}")
    print("=" * 70)

    cfg = DEFAULT_CONFIG
    model = MiniK3ForCausalLM(cfg).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    tokenizer = K3Tokenizer(args.tokenizer_model)

    print("[*] Model loaded successfully.")
    print("[*] Running sanity sample evaluation...")

    # Sample sanity problem
    prompt = "The sky is usually "
    choices = ["blue during the day.", "green made of cheese.", "made of stone.", "swimming in fish."]
    pred = score_multiple_choice(model, tokenizer, prompt, choices, device)
    print(f"    Prompt: '{prompt}'")
    for i, c in enumerate(choices):
        prefix = "-> " if i == pred else "   "
        print(f"    {prefix}[{i}] {c}")
    print(f"[*] Selection: Choice [{pred}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
