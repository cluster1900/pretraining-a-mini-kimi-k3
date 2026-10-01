"""Sequence fitting keeps the answer when a record is longer than the training length."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.alignment_fit import fit_preference, fit_sft


class FitTests(unittest.TestCase):
    def test_sft_tail_keeps_the_answer(self):
        ids = [1, 2, 3, 4, 5]
        labels = [-100, -100, -100, 4, 5]
        kept, kept_labels, dropped = fit_sft(ids, labels, 3)
        self.assertEqual(dropped, 2)
        self.assertEqual(kept, [3, 4, 5])
        self.assertEqual(kept_labels, [-100, 4, 5])

    def test_sft_without_a_target_is_rejected(self):
        self.assertIsNone(fit_sft([1, 2, 3], [-100, -100, -100], 2))

    def test_preference_keeps_one_shared_prompt_boundary(self):
        chosen, rejected, prompt_len = fit_preference(
            [1, 2, 3, 9], [1, 2, 3, 3, 8], 3, 3)
        self.assertEqual(chosen, [2, 3, 9])
        self.assertEqual(rejected, [2, 3, 3])
        self.assertEqual(prompt_len, 2)


if __name__ == "__main__":
    unittest.main()
