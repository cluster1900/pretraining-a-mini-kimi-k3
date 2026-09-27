"""
Standalone Causal Multiple-Choice Benchmark Evaluator for Mini Kimi K3.
Features:
- Right-padding (critical for causal models; left-padding corrupts context)
- Length-normalized log-probability scoring: sum(log P(choice_tokens)) / len(choice_tokens)
- Independent read-only evaluation from checkpoint volume
"""

import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import List
import torch
import torch.nn.functional as F

from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.data.tokenizer import K3Tokenizer
from train.eval_answers import resolve_choice_index


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mini K3 Checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (must contain model.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_model", default=None, help="Directory containing tiktoken.model. Required for --problems and --sanity.")
    parser.add_argument("--manifest", default=None, help="Validation manifest. Reports held-out next-token loss, the training evaluation.")
    parser.add_argument("--batches", type=int, default=32, help="Validation batches when --manifest is set.")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--problems", type=str, default=None, help="JSONL of {prompt, choices, answer}. answer is an index or the choice text.")
    parser.add_argument("--report", type=str, default=None, help="Where to write the accuracy report")
    parser.add_argument("--sanity", action="store_true", help="Score one fixed sentence. This is not a benchmark result.")
    return parser.parse_args()


def load_problems(path: str) -> List[dict]:
    problems = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        choices = row.get("choices")
        if not isinstance(row.get("prompt"), str) or not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"Line {line_no} needs a prompt and at least two choices")
        answer = resolve_choice_index(row.get("answer"), choices, line_no)
        problems.append({"prompt": row["prompt"], "choices": choices, "answer": answer})
    if not problems:
        raise ValueError(f"No problems in {path}")
    return problems


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
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
    print("[*] Model loaded successfully.")
    if args.manifest:
        from train.data.loader import MultiSourceDataLoader
        seq_len = args.sequence_length or cfg.sequence_length
        loader = MultiSourceDataLoader(manifest_path=args.manifest, seq_len=seq_len, batch_size=1)
        if not loader.streams:
            raise RuntimeError(f"Validation manifest has no readable shards: {args.manifest}")
        total = 0.0
        model.eval()
        amp = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
        with torch.no_grad():
            for _ in range(args.batches):
                batch, labels = loader.next_batch(device)
                with amp:
                    total += float(model(batch, labels=labels)["loss"])
        loss = total / args.batches
        report = {"status": "validation_loss", "batches": args.batches, "sequence_length": seq_len,
                  "loss": loss, "checkpoint": str(model_file)}
        print(f"[*] Validation loss over {args.batches} batches: {loss:.4f}")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return
    if not args.problems and not args.sanity:
        raise SystemExit("Pass --manifest for held-out loss, --problems for a question file, or --sanity for the sentence check.")
    if not args.tokenizer_model:
        raise SystemExit("--tokenizer_model is required for --problems and --sanity")
    tokenizer = K3Tokenizer(args.tokenizer_model)
    if args.problems:
        problems = load_problems(args.problems)
        correct = 0
        for item in problems:
            pred = score_multiple_choice(model, tokenizer, item["prompt"], item["choices"], device)
            correct += int(pred == item["answer"])
        accuracy = correct / len(problems)
        report = {"status": "scored", "problems": len(problems), "correct": correct, "accuracy": accuracy,
                  "checkpoint": str(model_file)}
        print(f"[*] Accuracy: {correct}/{len(problems)} = {accuracy:.4f}")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return
    if not args.sanity:
        raise SystemExit("Pass --problems JSONL for a scored evaluation, or --sanity for the single-sentence check.")
    prompt = "The sky is usually "
    choices = ["blue during the day.", "green made of cheese.", "made of stone.", "swimming in fish."]
    pred = score_multiple_choice(model, tokenizer, prompt, choices, device)
    print("[*] Sanity check only. This selection is not a benchmark score.")
    print(f"    Prompt: '{prompt}'")
    for i, c in enumerate(choices):
        prefix = "-> " if i == pred else "   "
        print(f"    {prefix}[{i}] {c}")
    print(f"[*] Selection: Choice [{pred}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
