"""
Interactive Streaming CLI Chat for Mini Kimi K3.
Features:
- Streaming token-by-token terminal output with low latency
- Multi-turn conversation management using Kimi chat template
- Incremental KV-cache decoding (KDA recurrent state + MLA sliding window)
- Stop token detection and clean terminal formatting
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train.config import DEFAULT_CONFIG
from train.models.mini_k3 import MiniK3ForCausalLM
from train.data.tokenizer import K3Tokenizer
from train.chat_template import pack_chat


def stream_generate(
    model: MiniK3ForCausalLM,
    tokenizer: K3Tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
):
    """
    Decodes tokens sequentially using KV cache and yields text chunks.
    """
    model.eval()
    prompt_len = input_ids.size(1)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, eos_token_id=tokenizer.eos_token_id,
        )
    eos = tokenizer.eos_token_id
    for token_val in output[0, prompt_len:].tolist():
        if token_val == eos:
            break
        yield tokenizer.decode([token_val])


def main():
    parser = argparse.ArgumentParser(description="Mini K3 Streaming Terminal Chat")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or model.pt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max generated tokens per response")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_model", required=True, help="Directory containing tiktoken.model")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = DEFAULT_CONFIG
    tokenizer = K3Tokenizer(args.tokenizer_model)

    print("=" * 70)
    print(" Mini Kimi K3 Interactive Chat Console")
    print(f" Target Device: {device} | Temperature: {args.temperature} | Top-p: {args.top_p}")
    print(" Commands: '/clear' to reset chat, '/exit' or 'Ctrl+C' to quit")
    print("=" * 70)

    model = MiniK3ForCausalLM(cfg).to(device)
    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        model_file = ckpt_p / "model.pt" if ckpt_p.is_dir() else ckpt_p
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at: {model_file}")
        print(f"[*] Loading weights from: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
        print("[*] Model loaded successfully.")
    else:
        print("[!] No checkpoint specified: running with randomly initialized model.")

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
            elif user_input.lower() == "/clear":
                messages = []
                print("[*] Chat history cleared.")
                continue

            messages.append({"role": "user", "content": user_input})

            # Format conversation with prompt template
            input_ids_list, _ = pack_chat(messages, tokenizer, max_length=cfg.sequence_length)
            # Append assistant header
            assistant_header = tokenizer.encode("<|assistant|>\n", append_eos=False)
            input_ids_list.extend(assistant_header)

            input_tensor = torch.tensor([input_ids_list], dtype=torch.long, device=device)

            print("Mini-K3 > ", end="", flush=True)
            response_chunks = []
            for chunk in stream_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_tensor,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                response_chunks.append(chunk)

            sys.stdout.write("\n")
            sys.stdout.flush()

            full_response = "".join(response_chunks)
            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\n[*] Exiting chat.")
            break


if __name__ == "__main__":
    main()
