"""Runtime regressions for the 2026-10-03 memory/throughput rewrite.

Covers, on CPU with tiny configs:
1. Block-dispatched routed experts == a naive per-token expert loop (values + grads).
2. Chunked LM loss == full-logit loss (values + grads), including ignored labels.
3. Batched/per-head Muon == per-matrix Newton-Schulz, with the 0.2*sqrt(max(m, n)) scale.
4. CSA2 MLA == a naive per-query reference (local band + indexer top-k entries),
   for single-block and many-block query tiling.
5. Activation checkpointing does not change loss or gradients (incl. CSA indexer
   aux loss), and every text-path parameter receives a gradient.
"""
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from train.config import MiniK3Config
from train.engine.muon import newton_schulz, orthogonalize
from train.kernels.situ_fused import _situ_forward_pytorch
from train.models import mla as mla_module
from train.models.mini_k3 import MiniK3ForCausalLM
from train.models.mla import MultiHeadLatentAttention
from train.models.moe import RoutedExperts


def tiny_config(**overrides) -> MiniK3Config:
    """6 layers: KDA, MLA(encoder, full), KDA, KDA, MLA(reindex), MLA(reuse)."""
    values = dict(
        hidden_size=64, num_layers=6, num_attention_heads=2, num_kda_heads=2, head_dim=32,
        mla_layers=[2, 5, 6], encoder_layers=3, num_routed_experts=8, top_k=2,
        moe_intermediate_size=32, routed_expert_hidden_size=32, q_lora_rank=32,
        kv_lora_rank=16, qk_nope_head_dim=16, v_head_dim=16, vocab_size=256,
        engram_layers=[2], engram_table_size=31, engram_heads=2, engram_head_dim=8,
        vision_layers=1, vision_hidden=32, vision_heads=4, csa_local=8, csa_group=4,
        csa_top_k=3, csa_index_dim=8, kv_cache_fp4=False, moe_block_size=4,
        loss_chunk_size=7, kda_chunk_size=8,
    )
    values.update(overrides)
    config = MiniK3Config(**values)
    config.validate()
    return config


def test_routed_experts_match_naive_loop():
    torch.manual_seed(0)
    experts = RoutedExperts(8, in_features=16, intermediate=12, beta=4.0, linear_beta=25.0,
                            init_std=0.2, block=4).double()
    x = torch.randn(37, 16, dtype=torch.float64, requires_grad=True)
    # Skewed routing: expert 0 takes most tokens, some experts take none.
    idx = torch.stack([torch.zeros(37, dtype=torch.long), torch.randint(1, 5, (37,))], dim=1)
    weight = torch.softmax(torch.randn(37, 2, dtype=torch.float64), dim=-1)
    got = experts(x, idx, weight)
    ref = torch.zeros_like(got)
    for t in range(37):
        for slot in range(2):
            e = int(idx[t, slot])
            hidden = _situ_forward_pytorch(experts.gate_up[e] @ x[t], 4.0, 25.0)
            ref[t] = ref[t] + weight[t, slot] * (experts.down[e] @ hidden)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6), (got - ref).abs().max()
    grads = torch.autograd.grad(got.square().sum(), (x, experts.gate_up, experts.down))
    refs = torch.autograd.grad(ref.square().sum(), (x, experts.gate_up, experts.down))
    for a, b in zip(grads, refs):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (a - b).abs().max()
    # Unused experts get exact zero gradients, not garbage from padded rows.
    assert grads[1][5:].abs().max() == 0 and grads[2][5:].abs().max() == 0


def test_chunked_loss_matches_full_logits():
    torch.manual_seed(1)
    model = MiniK3ForCausalLM(tiny_config(loss_chunk_size=5)).double()
    hidden = torch.randn(2, 23, 64, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 256, (2, 23))
    labels[0, :4] = -100
    labels[1, 10:] = -100
    chunked = model._chunked_ce(hidden, labels)
    full = model._mean_ce(model.lm_head(hidden), labels)
    assert torch.allclose(chunked, full, atol=1e-10), (chunked, full)
    g1 = torch.autograd.grad(chunked, (hidden, model.lm_head.weight))
    g2 = torch.autograd.grad(full, (hidden, model.lm_head.weight))
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-10)
    all_ignored = torch.full_like(labels, -100)
    assert float(model._chunked_ce(hidden, all_ignored)) == 0.0


def test_muon_batched_and_scaled():
    torch.manual_seed(2)
    stacked = torch.randn(5, 12, 7)
    batched = orthogonalize(stacked, 0, 5, batched=True, update_scale=0.2)
    for e in range(5):
        single = newton_schulz(stacked[e], 5) * (0.2 * math.sqrt(12))
        assert torch.allclose(batched[e], single, atol=1e-5)
    heads = torch.randn(4 * 6, 10)
    per_head = orthogonalize(heads, 4, 5, update_scale=0.2)
    for h in range(4):
        single = newton_schulz(heads[h * 6:(h + 1) * 6], 5) * (0.2 * math.sqrt(10))
        assert torch.allclose(per_head[h * 6:(h + 1) * 6], single, atol=1e-5)
    # Update RMS ~ 0.2 (Moonlight convention) for a square-ish matrix.
    rms = orthogonalize(torch.randn(64, 64), 0, 5, update_scale=0.2).pow(2).mean().sqrt()
    assert 0.12 < float(rms) < 0.25, float(rms)


def _naive_csa(layer: MultiHeadLatentAttention, hidden: torch.Tensor) -> torch.Tensor:
    cfg = layer.config
    batch, length, _ = hidden.shape
    heads, dim = layer.num_heads, layer.qk_nope_dim
    local, group, top_k = cfg.csa_local, cfg.csa_group, cfg.csa_top_k
    q = layer.q_up(layer.q_norm(layer.q_down(hidden))).view(batch, length, heads, dim).transpose(1, 2)
    lat = layer.kv_norm(layer.kv_down(hidden))
    k_raw, v_raw = layer._rebuild_kv(lat)
    groups = length // group
    entries = layer.entry_proj(lat[:, :groups * group].reshape(batch, groups, group * lat.shape[-1]))
    k_ent, v_ent = layer._rebuild_kv(entries)
    index = layer._index_linear(layer.index_q, hidden) @ layer._index_linear(layer.index_k, entries).transpose(-1, -2)
    index = index / math.sqrt(cfg.csa_index_dim)
    out = torch.zeros(batch, heads, length, layer.v_head_dim, dtype=hidden.dtype)
    for b in range(batch):
        for t in range(length):
            band = list(range(max(0, t - local + 1), t + 1))
            eligible = [g for g in range(groups) if g * group + group - 1 < t - local + 1]
            if eligible:
                scores = index[b, t, eligible]
                keep = scores.topk(min(top_k, len(eligible))).indices.tolist()
                chosen = [eligible[i] for i in keep]
            else:
                chosen = []
            for h in range(heads):
                logits = [q[b, h, t] @ k_raw[b, h, j] / math.sqrt(dim) for j in band]
                logits += [q[b, h, t] @ k_ent[b, h, g] / math.sqrt(dim) for g in chosen]
                w = torch.softmax(torch.stack(logits), dim=0)
                values = [v_raw[b, h, j] for j in band] + [v_ent[b, h, g] for g in chosen]
                out[b, h, t] = (w[:, None] * torch.stack(values)).sum(0)
    out = layer._gate(hidden, out)
    return layer.out_proj(out.transpose(1, 2).reshape(batch, length, -1))


def test_csa2_matches_naive_reference():
    torch.manual_seed(3)
    cfg = tiny_config()
    layer = MultiHeadLatentAttention(cfg).double().eval()
    hidden = torch.randn(2, 41, 64, dtype=torch.float64)
    ref = _naive_csa(layer, hidden)
    with torch.no_grad():
        for absorb in (False, True):
            layer._absorb_override = absorb
            got = layer(hidden)
            saved = mla_module._SCORE_BUDGET
            try:
                mla_module._SCORE_BUDGET = 1  # one query per block
                tiled = layer(hidden)
            finally:
                mla_module._SCORE_BUDGET = saved
            assert torch.allclose(got, ref, atol=1e-6), (absorb, (got - ref).abs().max())
            assert torch.allclose(tiled, ref, atol=1e-6), (absorb, (tiled - ref).abs().max())
        layer._absorb_override = None


def test_checkpointing_preserves_loss_and_grads():
    torch.manual_seed(4)
    ids = torch.randint(0, 256, (2, 45))
    results = []
    for ckpt in (False, True):
        torch.manual_seed(5)
        model = MiniK3ForCausalLM(tiny_config(activation_checkpointing=ckpt)).double().train()
        out = model(ids, labels=ids, compute_logits=False)
        assert out["aux_loss"] is not None and torch.isfinite(out["aux_loss"])
        expected = out["lm_loss"] + model.mtp_lambda * out["mtp_loss"] + model.config.csa_indexer_loss_weight * out["aux_loss"]
        assert torch.allclose(out["loss"], expected)
        out["loss"].backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        missing = [n for n, g in grads.items() if g is None and not n.startswith("vision.")]
        assert not missing, missing
        results.append((out["loss"].detach(), grads))
    (loss_a, grads_a), (loss_b, grads_b) = results
    assert torch.allclose(loss_a, loss_b, atol=1e-10), (loss_a, loss_b)
    for name, grad in grads_a.items():
        if grad is None:
            continue
        assert torch.allclose(grad, grads_b[name], atol=1e-8, rtol=1e-6), name


def test_logits_to_keep():
    torch.manual_seed(6)
    model = MiniK3ForCausalLM(tiny_config()).eval()
    ids = torch.randint(0, 256, (1, 19))
    with torch.no_grad():
        full = model(ids)["logits"]
        last = model(ids, logits_to_keep=1)["logits"]
    assert last.shape[1] == 1 and torch.allclose(full[:, -1:], last, atol=1e-6)


def main():
    test_routed_experts_match_naive_loop()
    test_chunked_loss_matches_full_logits()
    test_muon_batched_and_scaled()
    test_csa2_matches_naive_reference()
    test_checkpointing_preserves_loss_and_grads()
    test_logits_to_keep()
    print("MODEL_RUNTIME_TESTS_PASSED")


if __name__ == "__main__":
    main()
