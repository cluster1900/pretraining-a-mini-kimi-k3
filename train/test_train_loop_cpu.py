"""CPU end-to-end test of the train.py loop with a tiny stand-in model.

Checks that an interrupted + resumed run (stopping in the stable phase right
before the decay switch, or inside the decay phase) ends with bit-identical
weights to an uninterrupted run, that validation/logging/checkpointing work,
and that every forward uses ``compute_logits=False``. The 1.15B model is never
built here; readiness is stubbed (it has its own tests).

Run: ``.venv/bin/python -m pytest train/test_train_loop_cpu.py -q``
"""
import json
import math
import sys
import tempfile
import unittest
from array import array
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train.train as tt
from train.config import DEFAULT_CONFIG

VOCAB = DEFAULT_CONFIG.vocab_size  # ln(V) must sit inside the step-0 probe window


class TinyLM(nn.Module):
    """Embedding + zero-initialised head: step-0 loss is exactly ln(V)."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, 4)
        self.proj = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, cfg.vocab_size, bias=False)
        nn.init.zeros_(self.head.weight)
        self.calls = []

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        embed = self.embed.weight.numel()
        return {"total": total, "active": total, "embed": embed, "non_embed_active": total - embed}

    def forward(self, x, labels=None, compute_logits=True):
        self.calls.append(compute_logits)
        h = self.proj(self.embed(x))
        logits = self.head(h).float()
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        mtp = F.cross_entropy(logits[:, :-2].reshape(-1, logits.shape[-1]), labels[:, 2:].reshape(-1))
        return {"loss": lm + 0.3 * mtp, "lm_loss": lm, "mtp_loss": mtp,
                "logits": logits if compute_logits else None}


def _manifest(root: Path, name: str, weights: dict, tokens: int) -> Path:
    g = torch.Generator().manual_seed(len(name))
    sources = {}
    for i, (src, weight) in enumerate(weights.items()):
        shard = root / f"{name}-{i}.bin"
        values = torch.randint(0, VOCAB, (tokens,), generator=g).tolist()
        with shard.open("wb") as handle:
            array("I", values).tofile(handle)
        sources[src] = {"weight": weight, "shards": [str(shard)],
                        "shard_metadata": [{"path": str(shard), "tokens": tokens, "bytes": 4 * tokens}]}
    path = root / f"{name}.json"
    path.write_text(json.dumps({"sources": sources}))
    return path


class Interrupt(Exception):
    pass


class TrainLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = replace(DEFAULT_CONFIG, sequence_length=8, gradient_accumulation_steps=2)
        self.train_manifest = _manifest(self.root, "train", self.cfg.stable_mix, 600)
        self.val_manifest = _manifest(self.root, "val", {"fineweb-edu": 1.0}, 200)
        self.models = []

    def _patches(self):
        cfg = self.cfg

        def build(_cfg):
            model = TinyLM(_cfg)
            self.models.append(model)
            return model

        readiness = {"manifest": {"sha256": "t"}, "validation": {"sha256": "v"}}
        return [
            mock.patch.object(tt, "DEFAULT_CONFIG", cfg),
            mock.patch.object(tt, "MiniK3ForCausalLM", build),
            mock.patch.object(tt, "check_training_readiness", lambda *a, **k: readiness),
        ]

    def _run(self, ckpt, extra=(), interrupt_after=None):
        argv = ["--data_manifest", str(self.train_manifest), "--validation_manifest", str(self.val_manifest),
                "--checkpoint_dir", str(ckpt), "--total_steps", "8", "--schedule_total_steps", "8",
                "--log_interval", "2", "--validation_interval", "4", "--validation_batches", "2", *extra]
        patches = self._patches()
        real_save = tt.CheckpointManager.save

        def save(mgr, step, *a, **k):
            path = real_save(mgr, step, *a, **k)
            if interrupt_after is not None and step == interrupt_after:
                raise Interrupt
            return path

        patches.append(mock.patch.object(tt.CheckpointManager, "save", save))
        for p in patches:
            p.start()
        try:
            tt.main(argv)
        except Interrupt:
            pass
        finally:
            for p in reversed(patches):
                p.stop()

    def _final(self, ckpt):
        return torch.load(Path(ckpt) / "step_000007" / "model.pt", weights_only=False)

    def test_uninterrupted_vs_resumed_runs_match(self):
        # decay starts at int(8 * 0.85) = 6
        self._run(self.root / "a", ["--save_interval", "3"])
        reference = self._final(self.root / "a")
        self.assertTrue(all(c is False for m in self.models for c in m.calls))
        meta = torch.load(self.root / "a" / "step_000007" / "meta.pt", weights_only=False)
        self.assertEqual(meta["phase"], "decay")
        self.assertIsNotNone(meta["val_lm_loss"])
        self.assertTrue(math.isfinite(meta["val_lm_loss"]))
        self.assertTrue((self.root / "a" / "best" / "COMPLETE").is_file())

        cases = {
            "inside_decay": (["--save_interval", "3"], 6),   # saved at 6 (decay), resume at 7
            "before_decay": (["--save_interval", "5"], 5),   # saved at 5 (stable), switch on resume
        }
        for name, (extra, stop) in cases.items():
            with self.subTest(name):
                ckpt = self.root / name
                self._run(ckpt, extra, interrupt_after=stop)
                self.assertFalse((ckpt / "step_000007").exists())
                self._run(ckpt, [*extra, "--resume"])
                resumed = self._final(ckpt)
                for key, value in reference.items():
                    self.assertTrue(torch.equal(value, resumed[key]), f"{name}: {key} differs after resume")

    def test_resume_refuses_changed_schedule(self):
        self._run(self.root / "c", ["--save_interval", "3"], interrupt_after=3)
        argv_extra = ["--save_interval", "3", "--resume"]
        argv = ["--data_manifest", str(self.train_manifest), "--validation_manifest", str(self.val_manifest),
                "--checkpoint_dir", str(self.root / "c"), "--total_steps", "8", "--schedule_total_steps", "9",
                *argv_extra]
        patches = self._patches()
        for p in patches:
            p.start()
        try:
            with self.assertRaisesRegex(ValueError, "schedule_total_steps"):
                tt.main(argv)
        finally:
            for p in reversed(patches):
                p.stop()


if __name__ == "__main__":
    unittest.main()
