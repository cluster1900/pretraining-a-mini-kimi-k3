"""Defaults stay on the 2048 Mini K3 run. CED and 1M-scale lengths are explicit."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.run_options import resolve_run, resolve_schedule_total_steps


class RunOptionTests(unittest.TestCase):
    def test_default_is_the_2048_mini_model(self):
        opts = resolve_run("mini-k3", None, False, 2048, 1_048_576)
        self.assertEqual(opts["sequence_length"], 2048)
        self.assertEqual(opts["peak_lr"], 6.0e-4)
        # The sliding MLA window was retired; it is no longer part of a run.
        self.assertNotIn("attention_window", opts)

    def test_continuation_lengths_require_a_checkpoint(self):
        for length in (4096, 8192, 16384):
            with self.assertRaises(ValueError):
                resolve_run("mini-k3", length, False, 2048, 1_048_576)
            opts = resolve_run(
                "mini-k3", length, False, 2048, 1_048_576,
                peak_lr=6.0e-5, init_checkpoint="step",
            )
            self.assertEqual(opts["sequence_length"], length)

    def test_million_token_training_requires_the_flag(self):
        with self.assertRaises(ValueError):
            resolve_run("mini-k3", 1_048_576, False, 2048, 1_048_576)
        opts = resolve_run(
            "mini-k3", 32768, True, 2048, 1_048_576,
            peak_lr=6.0e-5, init_checkpoint="step",
        )
        self.assertEqual(opts["sequence_length"], 32768)

    def test_continuation_refuses_the_pretrain_peak_by_default(self):
        with self.assertRaises(ValueError):
            resolve_run("mini-k3", 4096, False, 2048, 1_048_576, init_checkpoint="step")
        opts = resolve_run("mini-k3", 4096, False, 2048, 1_048_576, peak_lr=6.0e-5, init_checkpoint="step")
        self.assertEqual(opts["peak_lr"], 6.0e-5)

    def test_ced_is_selectable(self):
        opts = resolve_run("ced", 2048, False, 2048, 1_048_576)
        self.assertEqual(opts["model"], "ced")
        self.assertNotIn("attention_window", opts)
        with self.assertRaises(ValueError):
            resolve_run("other", 2048, False, 2048, 1_048_576)

    def test_schedule_length_defaults_to_run_length(self):
        self.assertEqual(resolve_schedule_total_steps(200), 200)
        self.assertEqual(resolve_schedule_total_steps(200, 38147), 38147)
        with self.assertRaises(ValueError):
            resolve_schedule_total_steps(200, 100)
        with self.assertRaises(ValueError):
            resolve_schedule_total_steps(0)


if __name__ == "__main__":
    unittest.main()
