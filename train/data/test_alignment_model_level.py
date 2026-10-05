"""Tiny-model checks for the alignment stack (never the 1.15B default on CPU)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F

from train.config import MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM

EOS = 5


def tiny_config(**overrides):
    values = dict(
        hidden_size=64, num_layers=4, num_attention_heads=4, num_kda_heads=2, head_dim=32,
        mla_layers=[2, 4], num_routed_experts=4, top_k=2, moe_intermediate_size=32,
        routed_expert_hidden_size=32, q_lora_rank=32, kv_lora_rank=32,
        qk_nope_head_dim=32, v_head_dim=32, vocab_size=128, engram_layers=[],
        vision_layers=1, vision_hidden=32, vision_heads=4, csa_local=4, csa_group=2, csa_top_k=2,
        kv_cache_fp4=False, encoder_layers=2, sequence_length=64, activation_checkpointing=False,
    )
    values.update(overrides)
    cfg = MiniK3Config(**values)
    cfg.validate()
    return cfg


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return MiniK3ForCausalLM(tiny_config())


class FakeTokenizer:
    eos_token_id = EOS
    pad_token_id = 0

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class TokenLogprobTests(unittest.TestCase):
    def test_chunked_logprobs_match_full_logits(self):
        from train.alignment import token_logprobs
        model = tiny_model().eval()
        ids = torch.randint(6, 128, (2, 11))
        with torch.no_grad():
            logits = model(ids)["logits"]
            expected = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            got = token_logprobs(model, ids, chunk_size=3)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max())

    def test_chunked_logprobs_backprop(self):
        from train.alignment import token_logprobs
        model = tiny_model().train()
        ids = torch.randint(6, 128, (1, 9))
        token_logprobs(model, ids, chunk_size=4).sum().backward()
        self.assertIsNotNone(model.lm_head.weight.grad)


class GrpoTests(unittest.TestCase):
    def test_rollout_stops_at_eos_and_masks_the_tail(self):
        from train.engine.muon import build_optimizer
        from train.rl_trainer import GRPOTrainer, prepare_frozen
        policy = tiny_model(1)
        ref = prepare_frozen(tiny_model(1), "cpu", fp16=False)
        # Bias the head toward EOS so some samples terminate early.
        with torch.no_grad():
            policy.lm_head.weight[EOS] += 0.5 * policy.lm_head.weight[EOS].sign()
        opt = build_optimizer([p for p in policy.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
        trainer = GRPOTrainer(policy, ref, opt, group_size=3, epochs=2, use_amp=False, micro_batch_size=2)
        torch.manual_seed(3)
        batch = trainer.rollout([[7, 8, 9], [10, 11]], ["x", "y"], FakeTokenizer(), max_new_tokens=6)
        self.assertTrue(bool(batch.finished.any()), "fixture must produce at least one early EOS")
        self.assertTrue(any(getattr(layer, "is_moe", False) for layer in policy.layers))
        for row, (p, n) in enumerate(zip(batch.prompt_lens, batch.lengths)):
            seq = batch.full_ids[row, :n].tolist()
            self.assertNotIn(EOS, seq[p:-1], "tokens after the first EOS must be dropped")
            self.assertEqual(batch.action_mask[row].sum().item(), n - p)
            self.assertTrue(torch.all(batch.action_mask[row, n - 1:] == 0))
            self.assertNotIn("5", batch.texts[row].split())
        metrics = trainer.update(batch)
        for key in ("loss", "kl", "reward_mean", "clip_frac"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])), key)
        self.assertTrue(all(layer.mlp.gate.expert_load.sum() == 0
                            for layer in policy.layers if getattr(layer, "is_moe", False)))


class PpoTests(unittest.TestCase):
    def test_rollout_and_multi_epoch_update(self):
        from train.alignment_models import RewardModel, ValueModel
        from train.engine.muon import build_optimizer
        from train.rl_trainer import PPOTrainer, prepare_frozen
        cfg = tiny_config()
        torch.manual_seed(2)
        policy = MiniK3ForCausalLM(cfg)
        ref = prepare_frozen(MiniK3ForCausalLM(cfg), "cpu", fp16=False)
        ref.load_state_dict(policy.state_dict())
        reward = prepare_frozen(RewardModel(cfg), "cpu", fp16=False)
        value = ValueModel(cfg)
        self.assertTrue(getattr(value.score.weight, "adam_only", False))
        trainer = PPOTrainer(
            policy, ref, reward, value,
            build_optimizer([p for p in policy.parameters() if p.requires_grad], lr=5e-3, weight_decay=0.0),
            build_optimizer([p for p in value.parameters() if p.requires_grad], lr=5e-3, weight_decay=0.0),
            ppo_epochs=2, use_amp=False, micro_batch_size=1,
        )
        rollout = trainer.generate_rollout([[7, 8, 9], [10, 11, 12, 13]], max_new_tokens=5, eos_token_id=EOS)
        self.assertEqual(rollout.old_logp.shape, rollout.action_mask.shape)
        self.assertTrue(torch.all(rollout.advantages[rollout.action_mask == 0] == 0))
        metrics = trainer.train_step(rollout)
        self.assertAlmostEqual(metrics["approx_kl_old_first_epoch"], 0.0, places=4)
        self.assertNotAlmostEqual(metrics["approx_kl_old"], 0.0, places=6)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))


class InferenceTests(unittest.TestCase):
    def test_stream_generate_matches_greedy_generate(self):
        from train.chat import stream_generate

        class Tok:
            eos_token_id = EOS
            def decode(self, ids):
                return "".join(f"<{i}>" for i in ids)

        model = tiny_model(4).eval()
        ids = torch.tensor([[9, 10, 11, 12]])
        with torch.no_grad():
            ref = model.generate(ids, max_new_tokens=6, temperature=0.0, eos_token_id=EOS)[0, 4:].tolist()
        if EOS in ref:
            ref = ref[: ref.index(EOS)]
        text = "".join(stream_generate(model, Tok(), ids, max_new_tokens=6, temperature=0.0, top_p=1.0))
        self.assertEqual(text, "".join(f"<{i}>" for i in ref))

    def test_chunked_prefill_matches_full_forward(self):
        from train.eval_long_context import prefill_chunked
        model = tiny_model(5).eval()
        ids = torch.randint(6, 128, (1, 23))
        with torch.no_grad():
            full = model(ids)["logits"][:, -1]
            chunked, _ = prefill_chunked(model, ids, chunk=8)
        self.assertTrue(torch.allclose(full, chunked, atol=1e-4), (full - chunked).abs().max())


class SupervisedLoopTests(unittest.TestCase):
    def test_partial_accumulation_and_update_logging(self):
        from train.alignment_train import run_accumulated
        from train.engine.muon import build_optimizer
        from train.rl_trainer import make_balancer, make_grad_scaler
        model = tiny_model(6).train()
        opt = build_optimizer([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
        logs = []

        def micro():
            ids = torch.randint(6, 128, (1, 10))
            out = model(ids, labels=ids, compute_logits=False)
            self.assertIsNone(out["logits"])
            return out["loss"], {"lm_loss": float(out["lm_loss"].detach())}

        history = run_accumulated(model, opt, make_grad_scaler("cpu"), make_balancer(model), steps=5, accum=2,
                                  micro_fn=micro, tag="SFT", log_interval=1, log=logs.append)
        self.assertEqual([h["micro_steps"] for h in history], [2, 4, 5])
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(layer.mlp.gate.expert_load.sum() == 0
                            for layer in model.layers if getattr(layer, "is_moe", False)))

    def test_scalar_heads_load_explicitly(self):
        from train.alignment_models import RewardModel, ValueModel
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            torch.save(MiniK3ForCausalLM(cfg).state_dict(), root / "model.pt")
            rm = RewardModel(cfg, checkpoint=root, source="causal")
            torch.save(rm.state_dict(), root / "reward_model.pt")
            with self.assertRaises(FileNotFoundError):
                ValueModel(cfg, checkpoint=root, source="scalar")
            with self.assertRaises(ValueError):
                ValueModel(cfg, checkpoint=root / "reward_model.pt", source="scalar")
            with self.assertRaises(ValueError):
                RewardModel(cfg, checkpoint=root / "reward_model.pt", source="causal")
            loaded = RewardModel(cfg, checkpoint=root, source="scalar")
            self.assertTrue(torch.equal(loaded.score.weight, rm.score.weight))
            value = ValueModel(cfg, checkpoint=root, source="causal")
            self.assertTrue(torch.equal(value.backbone.lm_head.weight, rm.backbone.lm_head.weight))


if __name__ == "__main__":
    unittest.main()
