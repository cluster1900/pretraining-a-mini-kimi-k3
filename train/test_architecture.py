"""Small checks for AttnRes, latent-norm order, and quantile balancing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train.config import MiniK3Config
from train.models.attn_res import AttentionResidual
from train.models.moe import KimiMoEBlock
from train.engine.balancer import NoAuxBalancer, bias_from_histogram


def main():
    torch.manual_seed(0)
    mix = AttentionResidual(8, 1e-6)
    sources = [torch.randn(2, 3, 8) for _ in range(4)]
    mixed = mix(sources)
    mean = torch.stack(sources, dim=2).mean(dim=2)
    assert torch.allclose(mixed, mean, atol=1e-5)

    hist = torch.zeros(2, 4)
    hist[0, 3] = 10
    hist[1, 0] = 10
    bias = bias_from_histogram(hist, 0.5, -2.0, 2.0)
    assert abs(float(bias.sum())) < 1e-5
    assert float(bias[0]) < float(bias[1])

    cfg = MiniK3Config(
        hidden_size=32, num_layers=2, num_attention_heads=2, num_kda_heads=2,
        head_dim=16, mla_layers=[2], num_routed_experts=8, top_k=2,
        moe_intermediate_size=16, routed_expert_hidden_size=16,
        q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16, v_head_dim=16,
    )
    block = KimiMoEBlock(cfg)
    tokens = torch.randn(16, 32)
    out = block(tokens.view(2, 8, 32))
    assert out.shape == (2, 8, 32) and torch.isfinite(out).all()
    assert block.gate._last_scores is not None
    block.gate.accumulate_margin_histogram()
    balancer = NoAuxBalancer(block)
    telemetry = balancer.step()
    assert abs(float(block.gate.e_score_correction_bias.mean())) < 1e-5
    assert block.gate.e_score_correction_bias.requires_grad is False
    assert 0.0 <= telemetry["dead_frac"] <= 1.0
    print("ARCHITECTURE_CHECKS_PASSED")


if __name__ == "__main__":
    main()
