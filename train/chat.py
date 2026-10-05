"""
Interactive Streaming CLI Chat for Mini Kimi K3.
Features:
- True streaming: one cached forward per new token (KDA recurrent state + MLA
  cache); text is printed as soon as it is stable
- Incremental UTF-8-safe decoding: the cumulative ids are decoded each step
  and only the new, complete text is emitted (an incomplete multi-byte
  character, shown by tiktoken as U+FFFD, is held back until it completes)
- Multi-turn conversation packed with ``chat_template.pack_chat``, which is
  token-identical to the audited SFT encoding (``tokenize_v2.encode_messages``);
  the oldest turns are dropped first when the history does not fit
- Checkpoint may be a run root (newest complete ``step_*``), a step directory,
  an alignment output directory or a ``model.pt``
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train.chat_template import pack_chat

REPLACEMENT = "\ufffd"


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample ``[B, 1]`` token ids from ``[B, V]`` logits (greedy when temperature <= 0)."""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        keep = (torch.cumsum(sorted_p, -1) - sorted_p) < top_p
        keep[..., 0] = True
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
        return sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
    return torch.multinomial(probs, 1)


class IncrementalDecoder:
    """Decode cumulative ids; emit only text that can no longer change."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        text = self.tokenizer.decode(self.ids)
        stable = text.rstrip(REPLACEMENT)
        return self._emit(stable)

    def flush(self) -> str:
        return self._emit(self.tokenizer.decode(self.ids))

    def _emit(self, text: str) -> str:
        if not text.startswith(self.emitted):
            # Decoding never rewrites earlier complete characters; guard anyway.
            common = 0
            for a, b in zip(text, self.emitted):
                if a != b:
                    break
                common += 1
            self.emitted = self.emitted[:common]
        new = text[len(self.emitted):]
        self.emitted = text
        return new


@torch.no_grad()
def stream_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = None,
) -> Iterator[str]:
    """Yield text chunks while sampling token by token with the KV cache."""
    model.eval()
    eos = tokenizer.eos_token_id
    cache = model.new_kv_cache()
    out = model(input_ids, use_cache=True, past_key_values=cache, logits_to_keep=1)
    decoder = IncrementalDecoder(tokenizer)
    for _ in range(max_new_tokens):
        next_id = sample_next(out["logits"][:, -1], temperature, top_p)
        token = int(next_id.item())
        if token == eos:
            break
        chunk = decoder.push(token)
        if chunk:
            yield chunk
        out = model(next_id, use_cache=True, past_key_values=cache)
    tail = decoder.flush()
    if tail:
        yield tail


def main():
    from train.alignment_models import load_causal_model
    from train.data.tokenizer import K3Tokenizer

    parser = argparse.ArgumentParser(description="Mini K3 Streaming Terminal Chat")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Run root (newest complete step_*), step dir, alignment dir or model.pt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max generated tokens per response")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Prompt+response budget. Default: the checkpoint's trained sequence length, else 2048")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_model", required=True, help="Directory containing tiktoken.model")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = K3Tokenizer(args.tokenizer_model)
    model, info, cfg = load_causal_model(args.checkpoint, device)
    model.eval()
    context = args.context_length or info.sequence_length or 2048
    if args.max_tokens >= context:
        raise ValueError("--max_tokens must be smaller than the context length")

    print("=" * 70)
    print(" Mini Kimi K3 Interactive Chat Console")
    print(f" Weights: {info.model_file}")
    print(f" Device: {device} | Temperature: {args.temperature} | Top-p: {args.top_p} | Context: {context}")
    print(" Commands: '/clear' to reset chat, '/exit' or 'Ctrl+C' to quit")
    print("=" * 70)

    messages: List[Dict[str, str]] = []
    while True:
        try:
            print("\nUser > ", end="", flush=True)
            user_input = sys.stdin.readline()
            if not user_input:
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit"):
                print("Bye!")
                break
            if user_input.lower() == "/clear":
                messages = []
                print("[*] Chat history cleared.")
                continue

            messages.append({"role": "user", "content": user_input})
            try:
                ids, _ = pack_chat(messages, tokenizer, max_length=context - args.max_tokens,
                                   add_generation_prompt=True)
            except ValueError as exc:
                messages.pop()
                print(f"[!] {exc}")
                continue
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

            print("Mini-K3 > ", end="", flush=True)
            chunks = []
            for chunk in stream_generate(model, tokenizer, input_tensor, args.max_tokens,
                                         args.temperature, args.top_p, device):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                chunks.append(chunk)
            sys.stdout.write("\n")
            sys.stdout.flush()
            response = "".join(chunks)
            # SFT bodies are encoded as content + "\n" + EOS; drop the newline
            # the model emits before EOS so re-packing reproduces that layout.
            if response.endswith("\n"):
                response = response[:-1]
            messages.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print("\n[*] Exiting chat.")
            break


if __name__ == "__main__":
    main()
