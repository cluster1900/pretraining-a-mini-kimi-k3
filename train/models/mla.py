"""
Gated Multi-Head Latent Attention.

K3 applies no positional encoding here. Position stays in the KDA decay.
The cache stores the KV latent and rebuilds per-head K and V. A full-rank
sigmoid gate scales the attention output before the output projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from train.config import MiniK3Config
from train.models.attention_mask import attention_blocked, window_key_start
from train.models.fp4 import PackedKV, dequantize_fp4, quantize_fp4


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: MiniK3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_up = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_nope_dim, bias=False)

        self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_up = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_dim + self.v_head_dim), bias=False)

        self.out_gate = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        group = config.csa_group
        self.entry_proj = nn.Linear(group * self.kv_lora_rank, self.kv_lora_rank, bias=False)
        index_dim = config.csa_index_dim
        self.index_q = nn.Linear(self.hidden_size, index_dim, bias=False)
        self.index_k = nn.Linear(self.kv_lora_rank, index_dim, bias=False)
        self.q_up.weight.muon_heads = self.num_heads
        self.kv_up.weight.muon_heads = self.num_heads
        self._published = None

    def _gate(self, hidden_states: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        b, length, _ = hidden_states.shape
        gate = torch.sigmoid(self.out_gate(hidden_states))
        gate = gate.view(b, length, self.num_heads, self.v_head_dim).transpose(1, 2)
        return attn_out * gate.to(attn_out.dtype)

    def _load_past(self, past_kv):
        if past_kv is None:
            return None
        if isinstance(past_kv, PackedKV):
            return dequantize_fp4(past_kv).to(self.kv_down.weight.dtype)
        return past_kv

    def _store(self, kv_lat: torch.Tensor):
        window = self.config.attention_window
        kept = kv_lat[:, -window:].detach()
        if self.config.kv_cache_fp4:
            return quantize_fp4(kept)
        return kept

    def _entries(self, latent: torch.Tensor, origin: int) -> torch.Tensor:
        """Complete groups aligned to absolute positions, not to the cache slice."""
        group = self.config.csa_group
        batch, tokens, rank = latent.shape
        start = ((origin + group - 1) // group) * group - origin
        groups = 0 if start >= tokens else (tokens - start) // group
        if groups <= 0:
            filler = latent.new_zeros(batch, 1, group * rank)
            if tokens:
                filler[:, 0, :rank] = latent[:, 0]
            return self.entry_proj(filler)[:, :0]
        chunk = latent[:, start:start + groups * group]
        return self.entry_proj(chunk.reshape(batch, groups, group * rank))

    def _dense_or_csa(self, hidden_states, q, kv_lat, cache_position, memory, mode):
        """Local tokens stay dense. Older context is compressed and indexed."""
        length = hidden_states.shape[1]
        origin = cache_position + length - kv_lat.shape[1]
        own_entries = self._entries(kv_lat, origin)
        entries = own_entries
        if mode in ("reindex", "reuse") and memory is not None:
            entries = memory["entries"]
        connect = own_entries.reshape(-1)[:1].sum() * 0
        connect = connect + self.index_q(hidden_states).reshape(-1)[:1].sum() * 0
        connect = connect + self.index_k(own_entries).reshape(-1)[:1].sum() * 0
        return self._compressed(hidden_states, q, kv_lat, entries, cache_position, memory, mode) + connect

    def _cached_dense(self, q, k, v, cache_position):
        window = self.config.attention_window
        length = q.shape[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        end_pos = cache_position + length - 1
        k_abs = torch.arange(end_pos - k.shape[2] + 1, end_pos + 1, device=q.device)[None, :]
        q_abs = torch.arange(cache_position, cache_position + length, device=q.device)[:, None]
        scores = scores.masked_fill(attention_blocked(q_abs, k_abs, window), float("-inf"))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, v)

    def _local_bank(self, q, k, v, cache_position):
        """One causal window of `csa_local` raw keys for every query."""
        local = self.config.csa_local
        batch, heads, length, dim = q.shape
        total = k.shape[2]
        pad = local - 1
        k_pad = F.pad(k, (0, 0, pad, 0))
        v_pad = F.pad(v, (0, 0, pad, 0))
        k_bank = k_pad.unfold(2, local, 1).permute(0, 1, 2, 4, 3)
        v_bank = v_pad.unfold(2, local, 1).permute(0, 1, 2, 4, 3)
        start = total - length
        k_bank = k_bank[:, :, start:start + length]
        v_bank = v_bank[:, :, start:start + length]
        scores = torch.einsum("bhld,bhlwd->bhlw", q, k_bank) / math.sqrt(dim)
        offset = torch.arange(local, device=q.device)
        kv_index = torch.arange(start, start + length, device=q.device)[:, None]
        source = kv_index - local + 1 + offset
        scores = scores.masked_fill(source[None, None] < 0, torch.finfo(scores.dtype).min)
        return scores, v_bank

    def _attend_gathered(self, values, chosen, weights):
        """Project the shared entries once, then gather per query in small chunks."""
        batch, heads, _, dim = values.shape
        length, n_take = chosen.shape[1], chosen.shape[2]
        outputs = []
        for start in range(0, length, 32):
            end = min(length, start + 32)
            take = chosen[:, start:end]
            index = take[:, None, :, :, None].expand(batch, heads, end - start, n_take, dim)
            source = values[:, :, None].expand(batch, heads, end - start, values.shape[2], dim)
            picked = torch.gather(source, 3, index)
            outputs.append(torch.einsum("bhlk,bhlkd->bhld", weights[:, :, start:end], picked))
        return torch.cat(outputs, dim=2)

    def _compressed(self, hidden_states, q, kv_lat, entries, cache_position, memory, mode):
        batch, heads, length, _ = q.shape
        group = self.config.csa_group
        local = self.config.csa_local
        usable = entries
        complete = usable.shape[1]
        k_raw, v_raw = self._rebuild_kv(kv_lat)
        local_scores, local_values = self._local_bank(q, k_raw, v_raw, cache_position)
        if complete == 0:
            weights = torch.softmax(local_scores.float(), dim=-1).to(q.dtype)
            self._published = {"entries": entries, "index": None}
            return torch.einsum("bhlw,bhlwd->bhld", weights, local_values)
        q_index = self.index_q(hidden_states)
        raw_scores = torch.matmul(q_index, self.index_k(usable).transpose(-1, -2))
        origin = cache_position + length - kv_lat.shape[1]
        start_group = (origin + group - 1) // group
        entry_last = (start_group + torch.arange(complete, device=q.device)) * group + (group - 1)
        query = torch.arange(cache_position, cache_position + length, device=q.device)
        eligible = entry_last.view(1, 1, -1) < (query - local + 1).clamp_min(0).view(1, -1, 1)
        eligible = eligible.expand(batch, -1, -1)
        masked = raw_scores.masked_fill(~eligible, torch.finfo(raw_scores.dtype).min)
        n_take = min(self.config.csa_top_k, usable.shape[1])
        chosen_scores, chosen = torch.topk(masked, k=n_take, dim=-1)
        if mode == "reuse" and memory is not None and memory.get("index") is not None:
            chosen = memory["index"]
            if chosen.shape[1] != length:
                chosen = chosen[:, cache_position:cache_position + length]
            chosen_scores = torch.gather(raw_scores, -1, chosen.clamp(0, complete - 1))
        _, all_values = self._rebuild_kv(usable)
        global_logits = chosen_scores.unsqueeze(1).expand(-1, heads, -1, -1)
        allowed = torch.gather(eligible, 2, chosen.clamp(0, complete - 1))
        global_logits = global_logits.masked_fill(~allowed.unsqueeze(1), torch.finfo(global_logits.dtype).min)
        scores = torch.cat((local_scores, global_logits.to(local_scores.dtype)), dim=-1)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        local_out = torch.einsum("bhlw,bhlwd->bhld", weights[..., :local], local_values)
        global_out = self._attend_gathered(all_values, chosen, weights[..., local:])
        self._published = {"entries": entries, "index": chosen}
        return local_out + global_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv=None, use_cache: bool = False, cache_position: int = 0,
        memory=None, mode: str = "full",
    ):
        del attention_mask
        b, l, _ = hidden_states.shape
        q_lat = self.q_norm(self.q_down(hidden_states))
        q = self.q_up(q_lat).view(b, l, self.num_heads, self.qk_nope_dim).transpose(1, 2)
        kv_lat = self.kv_norm(self.kv_down(hidden_states))
        past = self._load_past(past_kv)
        if past is not None:
            kv_lat = torch.cat([past, kv_lat], dim=1)[:, -self.config.attention_window:]
        attn_out = self._dense_or_csa(hidden_states, q, kv_lat, cache_position, memory, mode)
        attn_out = self._gate(hidden_states, attn_out)
        result = self.out_proj(attn_out.transpose(1, 2).contiguous().view(b, l, -1))
        if not use_cache:
            return result
        return result, self._store(kv_lat)

    def _windowed_attention(self, q, k, v, window):
        block = min(window, 512)
        outputs = []
        scale = 1.0 / math.sqrt(q.shape[-1])
        length = q.shape[2]
        for start in range(0, length, block):
            end = min(length, start + block)
            key_start = window_key_start(start, window)
            qs = q[:, :, start:end]
            ks = k[:, :, key_start:end]
            vs = v[:, :, key_start:end]
            scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
            q_pos = torch.arange(start, end, device=q.device)[:, None]
            k_pos = torch.arange(key_start, end, device=q.device)[None, :]
            scores = scores.masked_fill(attention_blocked(q_pos, k_pos, window), float("-inf"))
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append(torch.matmul(weights, vs))
        return torch.cat(outputs, dim=2)

    def _rebuild_kv(self, kv_lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, length, _ = kv_lat.shape
        kv_full = self.kv_up(kv_lat).view(b, length, self.num_heads, self.qk_nope_dim + self.v_head_dim)
        keys = kv_full[..., : self.qk_nope_dim].transpose(1, 2)
        values = kv_full[..., self.qk_nope_dim :].transpose(1, 2)
        return keys, values
