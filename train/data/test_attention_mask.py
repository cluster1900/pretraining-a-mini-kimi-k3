"""The sliding window must keep every query's legal keys, including the first query in a block."""
import importlib.util
import unittest
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "models" / "attention_mask.py"
_spec = importlib.util.spec_from_file_location("attention_mask", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
attention_blocked = _mod.attention_blocked
window_key_start = _mod.window_key_start


class WindowMaskTests(unittest.TestCase):
    def test_block_start_still_sees_a_full_window(self):
        window, block = 8, 4
        start = 10
        # Taking keys from the block end drops positions 3, 4 and 5 for query 10.
        self.assertEqual(max(0, start + block - window), 6)
        self.assertEqual(window_key_start(start, window), 3)
        visible = [
            k for k in range(window_key_start(start, window), start + block)
            if not attention_blocked(start, k, window)
        ]
        self.assertEqual(visible, list(range(3, 11)))

    def test_future_and_stale_keys_are_blocked(self):
        window = 4
        self.assertTrue(attention_blocked(5, 6, window))
        self.assertTrue(attention_blocked(5, 1, window))
        self.assertFalse(attention_blocked(5, 2, window))
        self.assertFalse(attention_blocked(5, 5, window))

    def test_training_length_inside_the_window_is_full_causal(self):
        window, length = 4096, 2048
        self.assertEqual(window_key_start(0, window), 0)
        for q in (0, 100, length - 1):
            blocked = [k for k in range(length) if attention_blocked(q, k, window)]
            self.assertEqual(blocked, list(range(q + 1, length)))


if __name__ == "__main__":
    unittest.main()
