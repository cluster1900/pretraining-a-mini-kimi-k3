"""Model-free checks for the alignment stack: GAE, KL, EOS handling, data binding, IO policy."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from train.alignment import k3_kl, masked_mean, response_logprob, token_clipped_surrogate
from train.alignment_fit import sft_prompt
from train.alignment_readiness import AlignmentReadinessError, check_alignment_jsonl, load_audited_gold_answers
from train.rl_trainer import action_mask_for, compute_gae, group_advantages, truncate_at_eos, whiten_masked
from train.rule_reward import rule_reward


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GaeTests(unittest.TestCase):
    def test_matches_reference_recursion(self):
        torch.manual_seed(0)
        rewards = torch.zeros(2, 6)
        values = torch.randn(2, 6)
        mask = torch.tensor([[0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]], dtype=torch.float32)
        rewards[0, 3] = 1.5
        rewards[1, 5] = -0.5
        gamma, lam = 0.9, 0.8
        adv, ret = compute_gae(rewards, values, mask, gamma, lam)
        for row, (lo, hi) in enumerate(((1, 4), (2, 6))):
            running = 0.0
            for t in reversed(range(lo, hi)):
                nxt = values[row, t + 1].item() if t + 1 < hi else 0.0
                delta = rewards[row, t].item() + gamma * nxt - values[row, t].item()
                running = delta + gamma * lam * running
                self.assertAlmostEqual(adv[row, t].item(), running, places=5)
                self.assertAlmostEqual(ret[row, t].item(), running + values[row, t].item(), places=5)
        self.assertTrue(torch.all(adv[mask == 0] == 0))

    def test_lambda_one_gamma_one_is_return_minus_value(self):
        values = torch.tensor([[0.2, 0.4, 0.1]])
        rewards = torch.tensor([[0.0, 0.0, 2.0]])
        mask = torch.ones(1, 3)
        adv, _ = compute_gae(rewards, values, mask, 1.0, 1.0)
        self.assertTrue(torch.allclose(adv, 2.0 - values))

    def test_whiten_masked_ignores_padding(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        w = whiten_masked(x, mask)
        self.assertAlmostEqual(masked_mean(w, mask).item(), 0.0, places=5)
        self.assertEqual(w[0, 3].item(), 0.0)


class KlAndSurrogateTests(unittest.TestCase):
    def test_k3_is_zero_at_equality_and_positive_with_gradient(self):
        new = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
        ref = torch.tensor([-1.0, -1.0, -3.0])
        kl = k3_kl(new, ref)
        self.assertEqual(kl[0].item(), 0.0)
        self.assertTrue(torch.all(kl[1:] > 0))
        kl.sum().backward()
        self.assertNotEqual(new.grad[1].item(), 0.0)

    def test_sequence_mean_difference_was_inert_but_k3_is_not(self):
        # Opposite per-token deviations cancel in mean(new - ref) yet the policy moved.
        new = torch.tensor([-1.0, -3.0])
        ref = torch.tensor([-2.0, -2.0])
        self.assertEqual((new - ref).mean().item(), 0.0)
        self.assertGreater(k3_kl(new, ref).mean().item(), 0.0)

    def test_clipping_engages_off_policy(self):
        new = torch.tensor([0.0, 0.0])
        old = torch.tensor([-1.0, 0.0])
        loss, ratio, hit = token_clipped_surrogate(new, old, torch.tensor([1.0, 1.0]), 0.2)
        self.assertAlmostEqual(loss[0].item(), -1.2, places=5)
        self.assertEqual(hit.tolist(), [1.0, 0.0])

    def test_length_norm(self):
        lp = torch.tensor([[-1.0, -2.0, -3.0]])
        m = torch.tensor([[0.0, 1.0, 1.0]])
        self.assertEqual(response_logprob(lp, m).item(), -5.0)
        self.assertEqual(response_logprob(lp, m, length_norm=True).item(), -2.5)


class EosAndPromptTests(unittest.TestCase):
    def test_truncate_after_first_eos(self):
        seq, done = truncate_at_eos([1, 2, 9, 7, 9, 4], prompt_len=2, eos_token_id=9)
        self.assertEqual(seq, [1, 2, 9])
        self.assertTrue(done)
        seq, done = truncate_at_eos([9, 2, 3], prompt_len=1, eos_token_id=9)
        self.assertEqual(seq, [9, 2, 3])
        self.assertFalse(done)

    def test_action_mask_covers_only_response_tokens(self):
        mask = action_mask_for([2, 3], [4, 5], 6, "cpu")
        self.assertEqual(mask.tolist(), [[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])

    def test_sft_prompt_never_contains_the_answer(self):
        ids = [10, 11, 12, 20, 21, 22, 5]
        labels = [-100, -100, -100, 20, 21, 22, 5]
        self.assertEqual(sft_prompt(ids, labels), [10, 11, 12])
        self.assertIsNone(sft_prompt([1, 2], [-100, -100]))

    def test_group_advantages(self):
        adv = group_advantages(torch.tensor([1.0, 0.0, 2.0, 2.0]), 2)
        self.assertEqual(adv.tolist(), [1.0, -1.0, 0.0, 0.0])

    def test_rule_reward_boxed_answer(self):
        self.assertEqual(rule_reward("<think>w</think> so \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"), 2.0)
        self.assertEqual(rule_reward("<think>w</think> \\boxed{12}", "2"), 1.0)
        self.assertEqual(rule_reward("<think>w</think> 4[EOS]junk", "4"), 1.0)


class ReadinessTests(unittest.TestCase):
    def _fixture(self, root: Path, answer_rows):
        decon = root / "decontaminated" / "openr1"
        decon.mkdir(parents=True)
        part = decon / "part-00000.jsonl"
        part.write_text("".join(json.dumps(r) + "\n" for r in answer_rows), encoding="utf-8")
        marker = decon / "COMPLETE.json"
        marker.write_text(json.dumps({"status": "complete", "parts": [
            {"path": str(part), "bytes": part.stat().st_size, "sha256": sha(part)}]}))
        tok = root / "tokenized" / "openr1"
        tok.mkdir(parents=True)
        data = tok / "train.jsonl"
        data.write_text(json.dumps({"id": "a", "split": "train", "input_ids": [1, 2, 3],
                                    "labels": [-100, -100, 3]}) + "\n", encoding="utf-8")
        pref = root / "pref.jsonl"
        pref.write_text('{"prompt_ids":[1],"chosen_ids":[1,2],"rejected_ids":[1,3],"prompt_len":1}\n')
        manifest = root / "alignment.json"
        manifest.write_text(json.dumps({
            "metadata": {"schema_version": 2, "status": "audited", "vocab_size": 8, "tokenizer_fingerprint": "fp"},
            "sources": {
                "openr1": {"status": "complete", "kind": "sft", "tokenizer_fingerprint": "fp",
                           "upstream_sha256": sha(marker),
                           "files": [{"path": str(data), "split": "train",
                                      "bytes": data.stat().st_size, "sha256": sha(data)}]},
                "ultrafeedback": {"status": "complete", "kind": "preference", "tokenizer_fingerprint": "fp",
                                  "files": [{"path": str(pref), "split": "train",
                                             "bytes": pref.stat().st_size, "sha256": sha(pref)}]},
            }}))
        return manifest, data, pref, part

    def test_mode_kinds(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, data, pref, _ = self._fixture(Path(td), [{"id": "a", "split": "train", "answer": "3"}])
            self.assertEqual(check_alignment_jsonl(manifest, data, "grpo", 8)["kind"], "sft")
            self.assertEqual(check_alignment_jsonl(manifest, data, "ppo", 8)["kind"], "sft")
            self.assertEqual(check_alignment_jsonl(manifest, pref, "ppo", 8)["kind"], "preference")
            for mode, path in (("grpo", pref), ("dpo", data), ("rm", data), ("sft", pref)):
                with self.assertRaises(AlignmentReadinessError):
                    check_alignment_jsonl(manifest, path, mode, 8)

    def test_gold_answers_are_bound_to_the_audited_chain(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [{"id": "a", "split": "train", "answer": "3"},
                    {"id": "b", "split": "validation", "answer": "4"},
                    {"id": "c", "split": "train"}]
            manifest, data, _, part = self._fixture(Path(td), rows)
            evidence = check_alignment_jsonl(manifest, data, "grpo", 8)
            answers, report = load_audited_gold_answers(evidence)
            self.assertEqual(answers, {"a": "3"})
            self.assertEqual(report["without_answer"], 1)
            part.write_text(part.read_text() + "\n")
            with self.assertRaises(AlignmentReadinessError):
                load_audited_gold_answers(evidence)

    def test_sources_without_answers_fail_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, data, _, _ = self._fixture(Path(td), [{"id": "a", "split": "train"}])
            evidence = check_alignment_jsonl(manifest, data, "grpo", 8)
            with self.assertRaises(AlignmentReadinessError):
                load_audited_gold_answers(evidence)


class OutputPolicyTests(unittest.TestCase):
    def test_output_dir_guard(self):
        from train.alignment_train import check_output_dir, default_output_dir
        self.assertEqual(default_output_dir("dpo").name, "alignment-dpo")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            check_output_dir(out, "sft", False)
            out.mkdir()
            (out / "reward_model.pt").write_bytes(b"x")
            (out / "alignment_meta.json").write_text('{"mode": "rm"}')
            with self.assertRaisesRegex(ValueError, "'rm'"):
                check_output_dir(out, "sft", False)
            check_output_dir(out, "sft", True)
            step = root / "run" / "step_000010"
            step.mkdir(parents=True)
            (step / "COMPLETE").write_text("")
            with self.assertRaises(ValueError):
                check_output_dir(root / "run", "sft", True)
            with self.assertRaises(ValueError):
                check_output_dir(step, "sft", True, input_paths=(step,))

    def test_latest_complete_step_and_trained_length(self):
        from train.alignment_models import resolve_checkpoint
        from train.eval_long_context import resolve_lengths
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(root)
            for step, complete in ((100, True), (200, True), (300, False)):
                d = root / f"step_{step:06d}"
                d.mkdir()
                torch.save({}, d / "model.pt")
                torch.save({"step": step, "run_signature": {"sequence_length": 4096, "model": "mini-k3"}},
                           d / "meta.pt")
                if complete:
                    (d / "COMPLETE").write_text("")
            info = resolve_checkpoint(root)
            self.assertEqual(info.directory.name, "step_000200")
            self.assertEqual(info.sequence_length, 4096)
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(root / "step_000300")
            self.assertEqual(resolve_lengths(None, 4096, False), [2048, 4096])
            self.assertEqual(resolve_lengths(None, None, False), [2048])
            with self.assertRaises(ValueError):
                resolve_lengths([8192], 4096, False)
            self.assertEqual(resolve_lengths([8192], 4096, True), [8192])


class IncrementalDecodeTests(unittest.TestCase):
    def test_multibyte_characters_are_never_split(self):
        from train.chat import IncrementalDecoder

        class ByteTokenizer:
            def decode(self, ids):
                return bytes(ids).decode("utf-8", errors="replace")

        text = "数学 ok é"
        decoder = IncrementalDecoder(ByteTokenizer())
        chunks = [decoder.push(b) for b in text.encode("utf-8")]
        chunks.append(decoder.flush())
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all("\ufffd" not in c for c in chunks))


if __name__ == "__main__":
    unittest.main()
