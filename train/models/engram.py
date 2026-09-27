"""Hashed N-gram memory. Two modules, orders 2/3/4, eight heads, prime buckets.

The output projection starts at zero, so a text batch still begins at the
uniform-vocabulary loss. Token ids are remapped through `compress_id` before
hashing; that map starts as the identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _primes(start: int, count: int) -> list[int]:
    found = []
    n = max(2, start)
    while len(found) < count:
        if all(n % p != 0 for p in range(2, int(n ** 0.5) + 1)):
            found.append(n)
        n += 1
    return found


class Engram(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.orders = tuple(range(2, config.engram_max_ngram + 1))
        self.heads = config.engram_heads
        self.head_dim = config.engram_head_dim
        self.vocab_size = config.vocab_size
        slots = len(self.orders) * self.heads
        primes = _primes(config.engram_table_size, slots)
        self.tables = nn.ModuleList(
            nn.Embedding(prime, self.head_dim) for prime in primes
        )
        for table in self.tables:
            table.weight.adam_only = True
        mem_dim = slots * self.head_dim
        self.conv = nn.Conv1d(mem_dim, mem_dim, kernel_size=4, groups=mem_dim, bias=False)
        self.gate = nn.Linear(config.hidden_size, mem_dim)
        self.out = nn.Linear(mem_dim, config.hidden_size, bias=False)
        generator = torch.Generator()
        generator.manual_seed(10007 * layer_id + 17)
        multipliers = torch.randint(0, 2**31 - 1, (slots, config.engram_max_ngram), generator=generator)
        multipliers = multipliers * 2 + 1
        self.register_buffer("multipliers", multipliers, persistent=True)
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.int64), persistent=True)
        self.register_buffer("compress_id", torch.arange(config.vocab_size, dtype=torch.int64), persistent=True)
        self.reset_structural_init()

    def reset_structural_init(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.out.weight)

    def _hashes(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = self.compress_id[token_ids.clamp(0, self.vocab_size - 1)]
        batch, length = ids.shape
        slots = self.primes.numel()
        hashes = []
        for order_index, order in enumerate(self.orders):
            for head in range(self.heads):
                slot = order_index * self.heads + head
                mixed = torch.zeros(batch, length, dtype=torch.int64, device=ids.device)
                for shift in range(order):
                    shifted = torch.zeros_like(ids) if shift == 0 else ids.roll(shifts=shift, dims=1)
                    if shift:
                        shifted = shifted.clone()
                        shifted[:, :shift] = 0
                    mixed = torch.bitwise_xor(mixed, shifted * int(self.multipliers[slot, shift]))
                hashes.append(torch.remainder(mixed, int(self.primes[slot])))
        return torch.stack(hashes, dim=-1)

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        hashes = self._hashes(token_ids)
        parts = [table(hashes[..., i]) for i, table in enumerate(self.tables)]
        memory = torch.cat(parts, dim=-1)
        mixture = F.pad(memory.transpose(1, 2), (3, 0))
        mixture = self.conv(mixture).transpose(1, 2)
        mixture = mixture[:, -hidden.shape[1]:]
        gated = torch.sigmoid(self.gate(hidden)) * mixture
        return self.out(gated)
