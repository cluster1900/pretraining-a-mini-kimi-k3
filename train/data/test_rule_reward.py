import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.rule_reward import rule_reward


class RuleRewardTests(unittest.TestCase):
    def test_format_and_answer(self):
        text = "<think>work</think> 42"
        self.assertEqual(rule_reward(text, "42"), 2.0)

    def test_missing_think_is_zero(self):
        self.assertEqual(rule_reward("just 42", "42"), 0.0)

    def test_format_without_gold_match(self):
        self.assertEqual(rule_reward("<think>work</think> 7", "42"), 1.0)


if __name__ == "__main__":
    unittest.main()
