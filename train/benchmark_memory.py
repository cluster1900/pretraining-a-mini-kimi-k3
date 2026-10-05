"""
Long-context cache memory benchmark for Mini Kimi K3 (CSA2).
Enforces Rule 8 of AGENTS.md:
"1M 推理能力必须同时通过无 cache/cache logits 等价测试、长文档评测和显存基准；仅提高位置上限或创建 cache 数据结构不视为完成。"

What the cache holds (2026-10-03 CSA2 layout):
- KDA layers: one [B, H, D, D] FP32 recurrent state + three [B, 512, 3] conv states. O(1).
- Encoder MLA layers (``full`` mode, 1-indexed 4 and 8): a raw-latent tail of
  max(csa_local, csa_group) tokens plus every compressed entry, one
  ``kv_lora_rank`` vector per ``csa_group`` tokens. O(L / csa_group).
- Decoder MLA layers (reindex/reuse, 12 and 13): only the raw tail; they read
  the encoder's entries. O(1).

So the cache is NOT O(1): at 1M tokens it is ~256 MiB in FP16 for the default
config (2 layers x 262,144 entries x 256 x 2 bytes), ~68 MiB with the opt-in FP4
store. Part 2 measures a small live model and checks the cache grows exactly
as predicted. Neither part is a trained-model 1M result.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM


def _mla_roles(cfg: MiniK3Config):
    encoder = [layer for layer in cfg.mla_layers if layer <= cfg.encoder_layers]
    decoder = [layer for layer in cfg.mla_layers if layer > cfg.encoder_layers]
    return encoder, decoder


def calculate_theoretical_cache_memory(cfg: MiniK3Config, batch_size: int = 1,
                                       seq_len: int = 1_048_576, latent_bytes: int = 2) -> dict:
    """Bytes held by the cache after ``seq_len`` tokens (latents stored at ``latent_bytes``)."""
    num_kda_layers = cfg.num_layers - len(cfg.mla_layers)
    proj = cfg.num_kda_heads * cfg.head_dim
    kda_state = num_kda_layers * batch_size * cfg.num_kda_heads * cfg.head_dim * cfg.head_dim * 4
    kda_conv = num_kda_layers * 3 * batch_size * proj * (cfg.short_conv_kernel_size - 1) * latent_bytes
    encoder, decoder = _mla_roles(cfg)
    tail_tokens = min(seq_len, max(cfg.csa_local, cfg.csa_group))
    tail = len(cfg.mla_layers) * batch_size * tail_tokens * cfg.kv_lora_rank * latent_bytes
    entries = seq_len // cfg.csa_group
    if cfg.kv_cache_fp4:
        entry_bytes = cfg.kv_lora_rank // 2 + 2  # packed nibbles + FP16 scale
    else:
        entry_bytes = cfg.kv_lora_rank * latent_bytes
    entry_total = len(encoder) * batch_size * entries * entry_bytes
    total = kda_state + kda_conv + tail + entry_total
    mib = 1024 ** 2
    return {
        "seq_len": seq_len,
        "entries_per_layer": entries,
        "kda_mb": (kda_state + kda_conv) / mib,
        "mla_tail_mb": tail / mib,
        "mla_entries_mb": entry_total / mib,
        "total_cache_mb": total / mib,
    }


def _live_cache_entries(cache, cfg):
    encoder, _ = _mla_roles(cfg)
    counts = []
    for layer in encoder:
        slot = cache.mla_keys[layer - 1]
        store = None if slot is None else slot["entries"]
        counts.append(0 if store is None else store.count)
    return counts


def run_memory_benchmark():
    print("=" * 80)
    print(" Mini Kimi K3: CSA2 cache memory benchmark (AGENTS.md rule 8)")
    print("=" * 80)
    cfg = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = _mla_roles(cfg)
    print(f"[*] Device: {device}")
    print(f"[*] {cfg.num_layers} layers: {cfg.num_layers - len(cfg.mla_layers)} KDA, "
          f"encoder MLA {encoder}, decoder MLA {decoder}")
    print(f"[*] CSA2: local {cfg.csa_local}, group {cfg.csa_group}, top-k {cfg.csa_top_k}, "
          f"kv_lora_rank {cfg.kv_lora_rank}, FP4 store {'on' if cfg.kv_cache_fp4 else 'off'}")

    print("\n[Part 1: theoretical cache footprint, batch 1, FP16 latents]")
    print("-" * 80)
    print(f"{'Seq len':>10} | {'entries/layer':>13} | {'KDA MB':>8} | {'tail MB':>8} | {'entries MB':>10} | {'total MB':>9}")
    print("-" * 80)
    for length in [2048, 4096, 8192, 32768, 131072, 524288, 1048576]:
        m = calculate_theoretical_cache_memory(cfg, 1, length)
        print(f"{length:10,d} | {m['entries_per_layer']:13,d} | {m['kda_mb']:8.2f} | "
              f"{m['mla_tail_mb']:8.2f} | {m['mla_entries_mb']:10.2f} | {m['total_cache_mb']:9.2f}")
    print("-" * 80)
    m_1m = calculate_theoretical_cache_memory(cfg, 1, 1_048_576)
    expected = len(encoder) * (1_048_576 // cfg.csa_group) * cfg.kv_lora_rank * 2 / 1024 ** 2
    assert abs(m_1m["mla_entries_mb"] - expected) < 1e-6
    print(f"[+] Cache grows linearly with L/csa_group: {m_1m['total_cache_mb']:.1f} MB at 1M (not O(1)).")

    print("\n[Part 2: live small model, cache growth matches the formula]")
    print("-" * 80)
    from train.test_model_runtime import tiny_config
    test_cfg = tiny_config()
    model = MiniK3ForCausalLM(test_cfg).to(device).eval()
    cache = model.new_kv_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated() / 1024 ** 2
    prompt = torch.randint(0, test_cfg.vocab_size, (1, 256), device=device)
    with torch.no_grad():
        model(prompt, use_cache=True, past_key_values=cache, logits_to_keep=1)
        for _ in range(128):
            token = torch.randint(0, test_cfg.vocab_size, (1, 1), device=device)
            model(token, use_cache=True, past_key_values=cache, logits_to_keep=1)
    counts = _live_cache_entries(cache, test_cfg)
    want = cache.position // test_cfg.csa_group
    print(f"[*] position {cache.position}: encoder entries per layer {counts} (expected {want})")
    assert all(count == want for count in counts), counts
    for layer in test_cfg.mla_layers:
        slot = cache.mla_keys[layer - 1]
        assert slot["tail"].shape[1] == max(test_cfg.csa_local, test_cfg.csa_group)
    print("    -> PASS: entries = L / csa_group on encoder layers; raw tails stay bounded.")
    if device.type == "cuda":
        mem_end = torch.cuda.memory_allocated() / 1024 ** 2
        mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"[*] CUDA MB: start {mem_start:.2f} | now {mem_end:.2f} | peak {mem_peak:.2f}")
    print("=" * 80)
    print(">>> Cache layout check passed for the small live model. This is not a 1M trained-model result;")
    print(">>> rule 8 also needs a trained checkpoint on eval_long_context.py and a full-size GPU memory run.")
    print("=" * 80)


if __name__ == "__main__":
    run_memory_benchmark()
