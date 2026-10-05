"""End-to-end CLI wiring of alignment_train.main for every mode on a tiny model (CPU)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from train.data.test_alignment_model_level import EOS, tiny_config
from train.models.mini_k3 import MiniK3ForCausalLM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeK3Tokenizer:
    eos_token_id = EOS
    pad_token_id = 0
    fingerprint = "fp"

    def __init__(self, *args):
        pass

    def decode(self, ids):
        return " ".join(map(str, ids))


def build_fixture(root: Path, cfg):
    torch.manual_seed(0)
    run = root / "pretrain"
    step = run / "step_000001"
    step.mkdir(parents=True)
    torch.save(MiniK3ForCausalLM(cfg).state_dict(), step / "model.pt")
    torch.save({"step": 1, "run_signature": {"model": "mini-k3", "sequence_length": 48,
                                             "attention_window": cfg.attention_window}}, step / "meta.pt")
    (step / "COMPLETE").write_text("step=1\n")

    decon = root / "data" / "decontaminated" / "openr1"
    decon.mkdir(parents=True)
    rows = [{"id": f"r{i}", "split": "train", "answer": str(i)} for i in range(4)]
    part = decon / "part-00000.jsonl"
    part.write_text("".join(json.dumps(r) + "\n" for r in rows))
    marker = decon / "COMPLETE.json"
    marker.write_text(json.dumps({"status": "complete", "parts": [
        {"path": str(part), "bytes": part.stat().st_size, "sha256": sha(part)}]}))

    tok = root / "data" / "tokenized" / "openr1"
    tok.mkdir(parents=True)
    sft = tok / "train.jsonl"
    with sft.open("w") as f:
        for i in range(4):
            prompt = [10 + i, 11, 12, 13]
            answer = [40 + i, 41, 42, EOS]
            f.write(json.dumps({"id": f"r{i}", "split": "train", "input_ids": prompt + answer,
                                "labels": [-100] * len(prompt) + answer}) + "\n")
    pref_dir = root / "data" / "tokenized" / "ultrafeedback"
    pref_dir.mkdir(parents=True)
    pref = pref_dir / "train.jsonl"
    with pref.open("w") as f:
        for i in range(4):
            prompt = [20 + i, 21, 22]
            f.write(json.dumps({"id": f"p{i}", "split": "train", "prompt_len": 3, "prompt_ids": prompt,
                                "chosen_ids": prompt + [50, 51, EOS],
                                "rejected_ids": prompt + [60, EOS]}) + "\n")
    manifest = root / "alignment.json"
    manifest.write_text(json.dumps({
        "metadata": {"schema_version": 2, "status": "audited", "vocab_size": cfg.vocab_size,
                     "tokenizer_fingerprint": "fp"},
        "sources": {
            "openr1": {"status": "complete", "kind": "sft", "tokenizer_fingerprint": "fp",
                       "upstream_sha256": sha(marker),
                       "files": [{"path": str(sft), "split": "train", "bytes": sft.stat().st_size,
                                  "sha256": sha(sft)}]},
            "ultrafeedback": {"status": "complete", "kind": "preference", "tokenizer_fingerprint": "fp",
                              "files": [{"path": str(pref), "split": "train", "bytes": pref.stat().st_size,
                                         "sha256": sha(pref)}]},
        }}))
    return run, sft, pref, manifest


class AlignmentCliTests(unittest.TestCase):
    def test_every_mode_runs_and_writes_separate_artifacts(self):
        import train.config
        from train import alignment_train
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as td, \
                patch.object(train.config, "DEFAULT_CONFIG", cfg), \
                patch("train.data.tokenizer.K3Tokenizer", FakeK3Tokenizer):
            root = Path(td)
            run, sft, pref, manifest = build_fixture(root, cfg)
            common = ["--alignment-manifest", str(manifest), "--device", "cpu", "--steps", "3",
                      "--grad_accum_steps", "2", "--log_interval", "1", "--eos_token_id", str(EOS),
                      "--pad_token_id", "0", "--max_new_tokens", "4", "--prompt_batch", "2"]
            out = lambda mode: str(root / f"alignment-{mode}")
            alignment_train.main(["--mode", "sft", "--checkpoint", str(run), "--jsonl", str(sft),
                                  "--output_dir", out("sft")] + common)
            meta = json.loads((root / "alignment-sft" / "alignment_meta.json").read_text())
            self.assertEqual(meta["mode"], "sft")
            self.assertEqual(meta["sequence_length"], 48)
            alignment_train.main(["--mode", "rm", "--checkpoint", str(run), "--jsonl", str(pref),
                                  "--output_dir", out("rm")] + common)
            alignment_train.main(["--mode", "dpo", "--checkpoint", out("sft"), "--jsonl", str(pref),
                                  "--output_dir", out("dpo"), "--dpo_length_norm"] + common)
            alignment_train.main(["--mode", "ppo", "--checkpoint", out("sft"), "--jsonl", str(pref),
                                  "--reward_checkpoint", out("rm"), "--output_dir", out("ppo")] + common + ["--steps", "1"])
            alignment_train.main(["--mode", "grpo", "--checkpoint", out("sft"), "--jsonl", str(sft),
                                  "--tokenizer_model", "unused", "--output_dir", out("grpo"),
                                  "--group_size", "2"] + common + ["--steps", "1"])
            self.assertTrue((root / "alignment-ppo" / "value_model.pt").is_file())
            self.assertTrue((root / "alignment-grpo" / "model.pt").is_file())
            with self.assertRaises(ValueError):
                alignment_train.main(["--mode", "dpo", "--checkpoint", out("sft"), "--jsonl", str(pref),
                                      "--output_dir", out("rm")] + common)
            with self.assertRaises(FileNotFoundError):
                # PPO must not accept a causal directory as a reward model.
                alignment_train.main(["--mode", "ppo", "--checkpoint", out("sft"), "--jsonl", str(pref),
                                      "--reward_checkpoint", out("sft"), "--output_dir", str(root / "x")] + common + ["--steps", "1"])


if __name__ == "__main__":
    unittest.main()
