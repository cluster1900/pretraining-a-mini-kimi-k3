"""
Cache Equivalence implementation checks.
Enforces Rule 8 of AGENTS.md:
"1M 推理能力必须同时通过无 cache/cache logits 等价测试、长文档评测和显存基准；仅提高位置上限或创建 cache 数据结构不视为完成。"

Tests:
1. KDA log-decay stays inside (g_min, 0); MLA has no RoPE width
2. End-to-end logits equivalence (max |diff| < 1e-4) between one no-cache forward
   and three cached schedules (prefill + decode, irregular chunked prefill, pure
   decode) on a model with encoder/reindex/reuse CSA2 layers, Engram, MoE and
   active top-k entry selection at 15x csa_local; cache holds L/csa_group entries
3. Greedy generation: cached decoding == prefix recomputation token for token

These are implementation checks only. They do not establish 1M capability.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.models.kda import bounded_log_decay


def test_position_comes_from_kda():
    print("\n" + "=" * 70)
    print("[Test 1/3] Verifying NoPE MLA and the KDA decay floor")
    print("=" * 70)
    assert DEFAULT_CONFIG.qk_rope_head_dim == 0
    z = torch.linspace(-8, 8, 9).view(1, 1, 1, 9)
    log_decay = bounded_log_decay(z, torch.zeros(1), -5.0)
    assert torch.isfinite(log_decay).all()
    assert float(log_decay.min()) > -5.0
    assert float(log_decay.max()) < 0.0
    saturated = bounded_log_decay(torch.tensor([1e6, -1e6]).view(1, 1, 1, 2), torch.zeros(1), -5.0)
    assert float(saturated.min()) >= -5.0
    assert float(saturated.max()) <= 0.0
    print(f"  -> log-decay range {float(log_decay.min()):.4f} .. {float(log_decay.max()):.4f} inside (-5, 0)")
    print(">>> Test 1 Passed: MLA has no RoPE width and KDA decay stays bounded! <<<\n")


def _cache_config():
    from train.test_model_runtime import tiny_config
    # csa_local 8, group 4, top-3: at 120 tokens there are 30 entries and the
    # indexer keeps 3 of up to 28 eligible, so selection is active.
    return tiny_config()


def _run_cached(model, ids, splits):
    cache = model.new_kv_cache()
    parts, start = [], 0
    with torch.no_grad():
        for size in splits:
            out = model(ids[:, start:start + size], use_cache=True, past_key_values=cache)
            parts.append(out["logits"])
            start += size
    assert start == ids.shape[1]
    return torch.cat(parts, dim=1), cache


def test_cache_logits_equivalence(device):
    print("=" * 70)
    print(f"[Test 2/3] No-cache vs cached logits (CSA2 encoder/reindex/reuse, Engram) on {device}")
    print("=" * 70)
    torch.manual_seed(42)
    cfg = _cache_config()
    model = MiniK3ForCausalLM(cfg).to(device).eval()
    length = 120
    input_ids = torch.randint(0, cfg.vocab_size, (1, length), device=device)
    with torch.no_grad():
        logits_full = model(input_ids)["logits"]
    schedules = {
        "prefill 37 + token decode": [37] + [1] * (length - 37),
        "chunked prefill 13/29/41/37": [13, 29, 41, 37],
        "token decode from 1": [1] * length,
    }
    tolerance = 1e-4
    for name, splits in schedules.items():
        logits_cached, cache = _run_cached(model, input_ids, splits)
        max_diff = (logits_full - logits_cached).abs().max().item()
        print(f"[*] {name:28s} max |diff| = {max_diff:.3e}")
        if max_diff >= tolerance:
            raise AssertionError(f"Cache equivalence failed for {name}: {max_diff} >= {tolerance}")
    # Cache size is O(L / csa_group) entries plus a bounded raw tail.
    encoder_layer = cfg.mla_layers[0] - 1
    entry = cache.mla_keys[encoder_layer]
    stored = entry["entries"].count
    assert stored == length // cfg.csa_group, stored
    assert entry["tail"].shape[1] == max(cfg.csa_local, cfg.csa_group)
    for layer_index in cfg.mla_layers[1:]:
        assert cache.mla_keys[layer_index - 1]["entries"] is None, "decoder MLA must not duplicate entries"
    print(f"    -> cache holds {stored} compressed entries + {entry['tail'].shape[1]} raw latents per encoder MLA layer")
    print(">>> Test 2 Passed <<<\n")


def test_generation_equivalence(device):
    print("=" * 70)
    print(f"[Test 3/3] Greedy generation: cached decode vs prefix recomputation on {device}")
    print("=" * 70)
    torch.manual_seed(7)
    cfg = _cache_config()
    model = MiniK3ForCausalLM(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 50), device=device)
    max_new_tokens = 12
    with torch.no_grad():
        gen_cached = model.generate(prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0)
        curr_ids = prompt.clone()
        for _ in range(max_new_tokens):
            next_id = model(curr_ids)["logits"][:, -1].argmax(-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=1)
    cached_tokens = gen_cached[0, 50:].tolist()
    recompute_tokens = curr_ids[0, 50:].tolist()
    print(f"[*] Cached:    {cached_tokens}")
    print(f"[*] Recompute: {recompute_tokens}")
    assert cached_tokens == recompute_tokens, "Token generation diverged"
    print(">>> Test 3 Passed <<<\n")


def main():
    print("=" * 80)
    print(" Mini Kimi K3: Cache Equivalence Suite (implementation level)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_position_comes_from_kda()
    test_cache_logits_equivalence(device)
    test_generation_equivalence(device)

    print("=" * 80)
    print(" CACHE CHECKS PASSED: implementation equivalence holds for this small model.")
    print(" 1M capability remains unverified until a trained checkpoint also passes the")
    print(" long-document retrieval and live memory benchmarks (AGENTS.md rule 8).")
    print("=" * 80)


if __name__ == "__main__":
    main()
