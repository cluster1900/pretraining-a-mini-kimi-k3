"""Checks for the modules added on top of the K3 backbone."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train.config import MiniK3Config
from train.models.fp4 import dequantize_fp4, quantize_fp4
from train.models.mhc import sinkhorn
from train.models.moonvit import MoonViT
from train.models.mini_k3 import MiniK3ForCausalLM
from train.engine.muon import build_optimizer


def main():
    torch.manual_seed(0)
    logits = torch.eye(4) * 6
    mix = sinkhorn(logits, 20)
    assert torch.allclose(mix.sum(-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(mix.sum(-2), torch.ones(4), atol=1e-4)

    values = torch.randn(2, 5, 8)
    restored = dequantize_fp4(quantize_fp4(values))
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 6
    assert torch.isfinite(restored).all()
    assert (restored - values).abs().max() <= scale.max() * 2

    cfg = MiniK3Config(
        hidden_size=64, num_layers=4, num_attention_heads=4, num_kda_heads=2, head_dim=32,
        mla_layers=[2, 4], num_routed_experts=4, top_k=2, moe_intermediate_size=32,
        routed_expert_hidden_size=32, q_lora_rank=32, kv_lora_rank=32,
        qk_nope_head_dim=32, v_head_dim=32, vocab_size=128, engram_layers=[2],
        engram_table_size=64, engram_heads=2, engram_head_dim=8, vision_layers=1,
        vision_hidden=32, vision_heads=4, csa_local=4, csa_group=2, csa_top_k=2,
        kv_cache_fp4=True, encoder_layers=2, moe_block_size=4,
    )
    cfg.validate()
    model = MiniK3ForCausalLM(cfg)
    assert any(not layer.is_mla for layer in model.layers)
    assert model.layers[-1].is_mla
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["lm_loss"])
    out["loss"].backward()
    assert model.engrams["2"].out.weight.grad is not None
    opt = build_optimizer(model.parameters(), lr=1e-3, weight_decay=0.0)
    opt.step()
    images = torch.randn(1, 3, 28, 28)
    visual = model.vision(images)
    assert visual.shape[0] == 1 and visual.shape[-1] == cfg.hidden_size
    assert torch.isfinite(visual).all()
    video = torch.randn(1, 2, 3, 28, 28)
    frames = MoonViT(cfg)(video)
    assert frames.shape[1] == 2 and torch.isfinite(frames).all()

    # FP4 cache is an opt-in approximation: it must run and stay close, not be exact.
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 30))
    with torch.no_grad():
        full = model(ids)["logits"]
        cache = model.new_kv_cache()
        parts = [model(ids[:, :11], use_cache=True, past_key_values=cache)["logits"]]
        for t in range(11, 30):
            parts.append(model(ids[:, t:t + 1], use_cache=True, past_key_values=cache)["logits"])
        cached = torch.cat(parts, dim=1)
    assert torch.isfinite(cached).all()
    assert (full - cached).abs().max() < 0.5 * full.abs().max()

    # Exact cache equivalence (FP4 off) lives in test_cache_equivalence.py.
    print("V41_MODULES_PASSED")


if __name__ == "__main__":
    main()
