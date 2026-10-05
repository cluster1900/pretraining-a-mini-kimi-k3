"""CPU tests for the pretraining engine: checkpoints, SpikeGuard, balancer, train.py CLI.

These never build the 1.15B model; they use nn.Linear / tiny fixtures.
Run: ``.venv/bin/python -m pytest train/test_training_engine.py -q``
"""
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.engine.balancer import bias_from_histogram, histogram_bin_values
from train.engine.checkpoint import CheckpointManager, CheckpointSafetyError
from train.engine.scheduler import is_in_decay_phase, wsd_lr
from train.engine.spike_guard import SpikeGuard


def _tiny():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return model, opt


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "ckpt"
        self.sig = {"model": "tiny", "world_size": 1}

    def _save(self, mgr, step, model, opt, guard, val=None, loader=None):
        extra = {"run_signature": self.sig, "total_tokens_seen": step * 10}
        if val is not None:
            extra["val_loss"] = val
        return mgr.save(step, model, opt, guard, data_loader_state=loader or {"train": {"cursor": step}},
                        extra_meta=extra, rank=0, world_size=1, scaler=torch.amp.GradScaler("cuda", enabled=False))

    def test_round_trip_restores_everything(self):
        model, opt = _tiny()
        guard = SpikeGuard()
        guard.check(1, 5.0); guard.record_ok(1)
        mgr = CheckpointManager(self.dir, keep_last_n=2)
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        path = self._save(mgr, 3, model, opt, guard, val=2.5)
        self.assertTrue((path / "COMPLETE").is_file())
        self.assertTrue((path / "optimizer.pt").is_file())
        self.assertTrue((path / "scaler.pt").is_file())
        self.assertFalse(any(path.glob("optimizer_rank*.pt")))
        expected_random = (random.random(), np.random.rand(), torch.rand(1).item())
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        opt_state = opt.state_dict()

        model2, opt2 = nn.Linear(4, 3), None
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
        guard2 = SpikeGuard()
        mgr2 = CheckpointManager(self.dir, keep_last_n=2)
        meta = mgr2.load_latest(model2, opt2, guard2, rank=0, world_size=1,
                                scaler=torch.amp.GradScaler("cuda", enabled=False),
                                expected_run_signature=self.sig)
        self.assertEqual(meta["step"], 3)
        self.assertEqual(meta["data_loader"], {"train": {"cursor": 3}})
        for k, v in model2.state_dict().items():
            self.assertTrue(torch.equal(v, weights[k]))
        self.assertTrue(torch.equal(opt2.state_dict()["state"][0]["exp_avg"], opt_state["state"][0]["exp_avg"]))
        self.assertEqual(guard2.ema_loss, 5.0)
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expected_random)
        self.assertEqual(mgr2.best_metric, 2.5)
        self.assertEqual(mgr2.best_step, 3)

    def test_best_metric_persists_across_restart(self):
        model, opt = _tiny()
        guard = SpikeGuard()
        mgr = CheckpointManager(self.dir, keep_last_n=5)
        self._save(mgr, 1, model, opt, guard, val=3.0)
        self._save(mgr, 2, model, opt, guard, val=2.0)
        best_inode = (self.dir / "best" / "model.pt").stat().st_ino
        self.assertEqual(best_inode, (self.dir / "step_000002" / "model.pt").stat().st_ino)  # hard link
        self._save(mgr, 3, model, opt, guard, val=2.5)  # worse: best stays at 2
        self.assertEqual(torch.load(self.dir / "best" / "meta.pt", weights_only=False)["step"], 2)

        restarted = CheckpointManager(self.dir, keep_last_n=5)
        restarted.load_latest(model, opt, guard, expected_run_signature=self.sig)
        self.assertEqual((restarted.best_metric, restarted.best_step), (2.0, 2))
        # A later 2.2 must NOT become best (it would if best_metric were lost on restart).
        self._save(restarted, 4, model, opt, guard, val=2.2)
        self.assertEqual(torch.load(self.dir / "best" / "meta.pt", weights_only=False)["step"], 2)
        self._save(restarted, 5, model, opt, guard, val=1.0)
        self.assertEqual(torch.load(self.dir / "best" / "meta.pt", weights_only=False)["step"], 5)

    def test_corrupt_newest_falls_back_to_previous(self):
        model, opt = _tiny()
        guard = SpikeGuard()
        mgr = CheckpointManager(self.dir, keep_last_n=5)
        self._save(mgr, 1, model, opt, guard)
        with torch.no_grad():
            model.weight.add_(1.0)
        self._save(mgr, 2, model, opt, guard)
        (self.dir / "step_000002" / "model.pt").write_bytes(b"truncated")
        fresh = nn.Linear(4, 3)
        with self.assertWarns(UserWarning):
            meta = CheckpointManager(self.dir).load_latest(
                fresh, torch.optim.AdamW(fresh.parameters()), SpikeGuard(), expected_run_signature=self.sig)
        self.assertEqual(meta["step"], 1)
        self.assertFalse(torch.equal(fresh.weight, model.weight))  # step-1 weights, not step-2

    def test_incomplete_directory_is_ignored(self):
        model, opt = _tiny()
        mgr = CheckpointManager(self.dir)
        self._save(mgr, 1, model, opt, SpikeGuard())
        shutil.copytree(self.dir / "step_000001", self.dir / "step_000002")
        (self.dir / "step_000002" / "COMPLETE").unlink()
        meta = CheckpointManager(self.dir).load_latest(model, opt, SpikeGuard(), expected_run_signature=self.sig)
        self.assertEqual(meta["step"], 1)

    def test_signature_and_world_size_mismatch_raise_without_fallback(self):
        model, opt = _tiny()
        mgr = CheckpointManager(self.dir)
        self._save(mgr, 1, model, opt, SpikeGuard())
        with self.assertRaisesRegex(CheckpointSafetyError, "run signature"):
            mgr.load_latest(model, opt, SpikeGuard(), expected_run_signature={"model": "other", "world_size": 1})
        with self.assertRaisesRegex(CheckpointSafetyError, "world_size"):
            mgr.load_latest(model, opt, SpikeGuard(), world_size=4, expected_run_signature=self.sig)

    def test_legacy_per_rank_optimizer_files_still_load(self):
        model, opt = _tiny()
        mgr = CheckpointManager(self.dir)
        path = self._save(mgr, 1, model, opt, SpikeGuard())
        (path / "optimizer.pt").rename(path / "optimizer_rank0.pt")
        (path / "scaler.pt").rename(path / "scaler_rank0.pt")
        meta = CheckpointManager(self.dir).load_latest(
            model, opt, SpikeGuard(), scaler=torch.amp.GradScaler("cuda", enabled=False),
            expected_run_signature=self.sig)
        self.assertEqual(meta["step"], 1)

    def test_resave_same_step_and_pruning(self):
        model, opt = _tiny()
        mgr = CheckpointManager(self.dir, keep_last_n=2, milestone_interval=1000)
        for step in (1, 2, 3, 3):
            self._save(mgr, step, model, opt, SpikeGuard())
        names = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(names, ["step_000002", "step_000003"])


class SpikeGuardTests(unittest.TestCase):
    def test_skip_counted_once_per_step(self):
        g = SpikeGuard()
        skip, _ = g.check(0, float("nan"))
        self.assertTrue(skip)
        g.record_skip("non-finite gradient", 0)  # same step: not counted again
        self.assertEqual((g.total_skips, g.consecutive_skips), (1, 1))

    def test_gradient_overflows_accumulate_to_abort_threshold(self):
        g = SpikeGuard()
        for step in range(5):
            skip, _ = g.check(step, 12.0)  # finite loss
            self.assertFalse(skip)
            g.record_skip("non-finite gradient", step)
        self.assertEqual(g.consecutive_skips, 5)
        self.assertIsNone(g.ema_loss)  # skipped steps never enter the EMA
        g.check(5, 12.0); g.record_ok(5)
        self.assertEqual(g.consecutive_skips, 0)
        self.assertEqual(g.ema_loss, 12.0)

    def test_spike_detection_and_state(self):
        g = SpikeGuard(spike_factor=1.5, alpha=0.5)
        for step in range(12):
            g.check(step, 2.0); g.record_ok(step)
        skip, reason = g.check(12, 10.0)
        self.assertTrue(skip and "spike" in reason)
        g2 = SpikeGuard(); g2.load_state_dict(g.state_dict())
        self.assertEqual((g2.ema_loss, g2.total_skips, g2.consecutive_skips), (2.0, 1, 1))


class BalancerTests(unittest.TestCase):
    def test_reader_matches_moe_writer_convention(self):
        lo, hi, bins = -2.0, 2.0, 256
        values = histogram_bin_values(bins, lo, hi)
        self.assertAlmostEqual(values[0].item(), lo)
        self.assertAlmostEqual(values[-1].item(), hi, places=5)
        # Write margins exactly like KimiGate.accumulate_margin_histogram.
        torch.manual_seed(0)
        margins = torch.stack([torch.randn(20000) * 0.3 - 0.1, torch.randn(20000) * 0.3 + 0.2], dim=1)
        scaled = (margins.clamp(lo, hi) - lo) / (hi - lo)
        idx = (scaled * (bins - 1)).round().long().clamp(0, bins - 1)
        hist = torch.zeros(2, bins)
        for e in range(2):
            hist[e] = torch.bincount(idx[:, e], minlength=bins).float()
        for level in (0.5, 0.9, 0.977):
            q = torch.quantile(margins, level, dim=0)
            bias = bias_from_histogram(hist, level, lo, hi)
            expected = -q - (-q).mean()
            self.assertTrue(torch.allclose(bias, expected, atol=(hi - lo) / (bins - 1)),
                            f"level {level}: {bias} vs {expected}")
        # An exactly representable margin is read back exactly (no half-bin shift).
        hist = torch.zeros(2, bins)
        hist[0, 0] = 1  # margin == lo
        hist[1, bins - 1] = 1  # margin == hi
        bias = bias_from_histogram(hist, 0.5, lo, hi)
        self.assertAlmostEqual((bias[0] - bias[1]).item(), hi - lo, places=5)


class TrainEntrypointTests(unittest.TestCase):
    def test_import_and_cli(self):
        import train.train as tt
        args = tt.parse_args([])
        self.assertEqual(args.validation_batches, 8)
        self.assertIsNone(args.schedule_total_steps)
        args = tt.parse_args(["--total_steps", "200", "--schedule_total_steps", "38147"])
        self.assertEqual(args.schedule_total_steps, 38147)
        with self.assertRaises(SystemExit):
            tt.parse_args(["--attention-window", "4096"])
        with self.assertRaises(SystemExit):
            tt.parse_args(["--validation_batches", "0"])

    def test_short_run_follows_10b_schedule_shape(self):
        peak = 6e-4
        warmup = max(1, int(38147 * 0.02))
        self.assertEqual(warmup, 762)
        self.assertAlmostEqual(wsd_lr(199, 38147, peak), peak * 200 / 762)
        # A 200-step schedule instead: 4-step warmup, peak, then decay from step 170.
        self.assertAlmostEqual(wsd_lr(100, 200, peak), peak)
        self.assertTrue(is_in_decay_phase(199, 200))
        self.assertFalse(is_in_decay_phase(199, 38147))

    def test_run_signature_covers_training_knobs(self):
        import argparse
        import train.train as tt
        from train.config import MiniK3Config
        cfg = MiniK3Config()
        args = argparse.Namespace(total_steps=200, seed=42)
        readiness = {"manifest": {"sha256": "a"}, "validation": {"sha256": "b"}}
        sig = tt.build_run_signature({"model": "mini-k3"}, cfg, args, 4, readiness, {"total": 1}, 38147)
        for key in ("gradient_accumulation_steps", "micro_batch_size", "seed", "warmup_frac", "decay_frac",
                    "min_lr_frac", "stable_mix", "decay_mix", "muon_update_scale", "moe_block_size",
                    "loss_chunk_size", "schedule_total_steps", "config_sha256"):
            self.assertIn(key, sig)
        self.assertNotIn("attention_window", sig)
        cfg.gradient_accumulation_steps = 16
        sig2 = tt.build_run_signature({"model": "mini-k3"}, cfg, args, 4, readiness, {"total": 1}, 38147)
        self.assertNotEqual(sig, sig2)

    def test_validation_replays_fixed_slice(self):
        import json
        from array import array
        import train.train as tt
        from train.data.loader import MultiSourceDataLoader

        class Echo(nn.Module):
            """lm_loss = mean token id, so the scored tokens are visible."""
            def forward(self, x, labels=None, compute_logits=True):
                assert compute_logits is False
                lm = x.float().mean()
                return {"loss": lm, "lm_loss": lm, "mtp_loss": lm * 0 + 1.0, "logits": None}

        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "v.bin"
            with shard.open("wb") as handle:
                array("I", range(64)).tofile(handle)
            manifest = Path(td) / "validation.json"
            manifest.write_text(json.dumps({"sources": {"v": {
                "weight": 1.0, "shards": [str(shard)],
                "shard_metadata": [{"path": str(shard), "tokens": 64, "bytes": 256}]}}}))
            loader = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            snap = loader.state_dict()
            model = Echo().train()
            device = torch.device("cpu")
            results = [tt.run_validation(model, loader, snap, 3, device, torch.float16, False, False, 1)
                       for _ in range(3)]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
            self.assertAlmostEqual(results[0][0], (1.5 + 5.5 + 9.5) / 3)
            self.assertEqual(results[0][1], 1.0)
            self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
