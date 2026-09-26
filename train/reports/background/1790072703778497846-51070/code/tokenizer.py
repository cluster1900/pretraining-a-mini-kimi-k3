"""Kimi BPE wrapper reconstructed from the pinned upstream tokenizer files.

Raw document text always uses ordinary encoding. Structural tokens must be
introduced explicitly by the caller. Missing/mismatched assets are fatal.
"""
import ast
import hashlib
import json
import os
from pathlib import Path


def upstream_pattern(path):
    """Read the literal pat_str definition without importing upstream code."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == 'TikTokenTokenizer':
            for node in cls.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pat_str' for t in node.targets):
                    value = node.value
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and value.func.attr == 'join' and isinstance(value.func.value, ast.Constant)
                        and value.func.value.value == '|' and len(value.args) == 1):
                        parts = ast.literal_eval(value.args[0])
                        if isinstance(parts, list) and all(isinstance(x, str) for x in parts):
                            return '|'.join(parts)
    raise ValueError('Unrecognized upstream pat_str; review tokenizer source before use')


def safe_pieces(text):
    """Match upstream 400k / 25k consecutive-whitespace splitting exactly."""
    for start in range(0, len(text), 400_000):
        s = text[start:start + 400_000]
        run = 0
        space = s[0].isspace() if s else False
        begin = 0
        for i, char in enumerate(s):
            current = char.isspace()
            if space != current:
                run, space = 1, current
            else:
                run += 1
                if run > 25_000:
                    yield s[begin:i]
                    begin, run = i, 1
        yield s[begin:]


class K3Tokenizer:
    VOCAB_SIZE = 163840
    BOS_TOKEN_ID = 163584
    EOS_TOKEN_ID = 163585
    NUM_RESERVED_SPECIAL_TOKENS = 256

    def __init__(self, model_path=None):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        self.model_path = Path(model_path or os.environ.get('MINIK3_TOKENIZER_DIR', '/data/mini-k3/data/tokenizer'))
        paths = [self.model_path / f for f in ('tiktoken.model', 'tokenizer_config.json', 'tokenization_kimi.py')]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f'Required tokenizer asset missing: {p}')
        self.asset_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        config = json.loads(paths[1].read_text())
        ranks = load_tiktoken_bpe(str(paths[0]))
        if len(ranks) != self.BOS_TOKEN_ID or set(ranks.values()) != set(range(self.BOS_TOKEN_ID)):
            raise ValueError('BPE ranks must cover exactly 0..163583')
        named = {int(k): v['content'] for k, v in config['added_tokens_decoder'].items()}
        if any(i < self.BOS_TOKEN_ID or i >= self.VOCAB_SIZE for i in named):
            raise ValueError('Special token ID outside reserved range')
        self.special_tokens = {named.get(i, f'<|reserved_token_{i}|>'): i
                               for i in range(self.BOS_TOKEN_ID, self.VOCAB_SIZE)}
        if len(self.special_tokens) != self.NUM_RESERVED_SPECIAL_TOKENS:
            raise ValueError('Duplicate special token spelling')
        for token, expected in (('[BOS]', self.BOS_TOKEN_ID), ('[EOS]', self.EOS_TOKEN_ID), ('[PAD]',163839)):
            if self.special_tokens.get(token) != expected:
                raise ValueError(f'Unexpected ID for {token}')
        self.pat_str = upstream_pattern(paths[2])
        self.enc = tiktoken.Encoding(name='mini-k3-pinned', pat_str=self.pat_str,
                                     mergeable_ranks=ranks, special_tokens=self.special_tokens)
        if self.enc.n_vocab != self.VOCAB_SIZE:
            raise ValueError('Unexpected tokenizer vocabulary size')
        self.vocab_size = self.VOCAB_SIZE
        self.bos_token_id = self.BOS_TOKEN_ID
        self.eos_token_id = self.EOS_TOKEN_ID
        self.pad_token_id = self.special_tokens['[PAD]']
        self.fingerprint = hashlib.sha256(json.dumps({'assets':self.asset_hashes,
            'algorithm':'upstream-safe-pieces-ordinary-v2'},sort_keys=True).encode()).hexdigest()

    def encode(self, text, append_eos=True):
        if not isinstance(text, str):
            raise TypeError('Tokenizer input must be a string')
        ids = [i for piece in safe_pieces(text) for i in self.enc.encode_ordinary(piece)]
        if append_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(self, texts, append_eos=True, num_threads=4):
        # Common short documents can use Rust workers without changing results.
        if all(isinstance(t,str) and len(t) < 25_000 for t in texts):
            result = self.enc.encode_ordinary_batch(texts, num_threads=num_threads)
            if append_eos:
                for ids in result: ids.append(self.eos_token_id)
            return result
        return [self.encode(t, append_eos=append_eos) for t in texts]

    def decode(self, tokens):
        return self.enc.decode(tokens)
