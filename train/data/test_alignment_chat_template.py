"""pack_chat must be token-identical to the audited tokenize_v2.encode_messages."""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tokenize_v2 import encode_messages  # read-only ground truth
from train.chat_template import pack_chat


class MergingTokenizer:
    """Greedy longest-match tokenizer whose vocabulary spans header/body boundaries.

    Encoding ``"<|user|>\\n" + content`` as one string merges across the
    boundary (e.g. ``">\\nH"``), so a template that does not encode header and
    body separately produces different ids than tokenize_v2.
    """
    eos_token_id = 5
    PIECES = ["|>\nH", ">\nH", ">\nW", "|>\n", "\nH", "<|", "|>", "user", "assistant", "system",
              "Hello", "World", "\n", " "]

    def __init__(self):
        self.vocab = {piece: 1000 + i for i, piece in enumerate(self.PIECES)}
        self.inverse = {v: k for k, v in self.vocab.items()}
        self.longest = max(len(p) for p in self.PIECES)

    def encode(self, text, append_eos=True):
        ids, i = [], 0
        while i < len(text):
            for size in range(min(self.longest, len(text) - i), 0, -1):
                piece = text[i:i + size]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    i += size
                    break
            else:
                ids.append(2000 + ord(text[i]))
                i += 1
        if append_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        out = []
        for t in ids:
            if t == self.eos_token_id:
                out.append("[EOS]")
            elif t in self.inverse:
                out.append(self.inverse[t])
            else:
                out.append(chr(t - 2000))
        return "".join(out)


CONVERSATIONS = [
    [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello there"},
     {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "What is 2+2?"},
     {"role": "assistant", "content": "<think>add</think> 4"}],
    [{"role": "user", "content": " leading space and\nnewline"}, {"role": "assistant", "content": "\nHello\n"}],
    [{"role": "user", "content": "数学 题目：1+1=?"}, {"role": "assistant", "content": "答案是 2。"}],
]


class ChatTemplateTests(unittest.TestCase):
    def setUp(self):
        self.tok = MergingTokenizer()

    def test_pack_chat_matches_tokenize_v2(self):
        for messages in CONVERSATIONS:
            expected = encode_messages(messages, self.tok)
            got = pack_chat(messages, self.tok, max_length=10_000)
            self.assertEqual(got, expected, messages)

    def test_boundary_merge_would_be_detected(self):
        joined = self.tok.encode("<|user|>\nHello\n", append_eos=False)
        split = self.tok.encode("<|user|>\n", append_eos=False) + self.tok.encode("Hello\n", append_eos=False)
        self.assertNotEqual(joined, split)

    def test_generation_prompt_is_the_sft_assistant_header(self):
        messages = CONVERSATIONS[1][:4]
        ids, labels = pack_chat(messages, self.tok, max_length=10_000, add_generation_prompt=True)
        full, _ = encode_messages(CONVERSATIONS[1], self.tok)
        self.assertEqual(ids, full[: len(ids)])
        self.assertTrue(all(label == -100 for label in labels[-3:]))

    def test_truncation_keeps_tail_and_newest_user_turn(self):
        messages = CONVERSATIONS[1]
        full, _ = encode_messages(messages, self.tok)
        last_two = encode_messages(messages[3:], self.tok)[0]
        limit = len(last_two) + 2
        ids, labels = pack_chat(messages, self.tok, max_length=limit)
        self.assertLessEqual(len(ids), limit)
        self.assertEqual(ids, last_two)
        self.assertEqual(ids, full[-len(ids):])
        self.assertEqual(len(ids), len(labels))

    def test_system_turn_is_kept_when_it_fits(self):
        messages = CONVERSATIONS[1]
        system = encode_messages([messages[0], {"role": "assistant", "content": "x"}], self.tok)[0]
        system = system[: len(self.tok.encode("<|system|>\n", False)) + len(self.tok.encode("Be brief.\n", False))]
        tail = encode_messages(messages[3:], self.tok)[0]
        ids, _ = pack_chat(messages, self.tok, max_length=len(system) + len(tail))
        self.assertEqual(ids, system + tail)

    def test_newest_user_turn_that_cannot_fit_raises(self):
        with self.assertRaises(ValueError):
            pack_chat(CONVERSATIONS[1], self.tok, max_length=5)

    @unittest.skipUnless(
        (Path(os.environ.get("MINIK3_TOKENIZER_DIR", "/data/mini-k3/data/tokenizer")) / "tiktoken.model").is_file(),
        "real K3 tokenizer assets not available locally")
    def test_real_tokenizer(self):
        from tokenizer import K3Tokenizer
        tok = K3Tokenizer()
        for messages in CONVERSATIONS:
            self.assertEqual(pack_chat(messages, tok, max_length=100_000), encode_messages(messages, tok))


if __name__ == "__main__":
    unittest.main()
