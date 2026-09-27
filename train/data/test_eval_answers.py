"""Letter labels such as A–D map onto choice positions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.eval_answers import resolve_choice_index


class AnswerLabelTests(unittest.TestCase):
    def test_letter_label_maps_to_choice_position(self):
        choices = ["Paris", "London", "Berlin", "Rome"]
        self.assertEqual(resolve_choice_index("B", choices, 1), 1)
        self.assertEqual(resolve_choice_index("d", choices, 2), 3)

    def test_choice_text_still_matches_directly(self):
        choices = ["Paris", "London"]
        self.assertEqual(resolve_choice_index("London", choices, 1), 1)

    def test_letter_past_the_last_choice_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_choice_index("E", ["Paris", "London", "Berlin", "Rome"], 4)


if __name__ == "__main__":
    unittest.main()
