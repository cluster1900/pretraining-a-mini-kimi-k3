"""Gated Multi-Head Latent Attention with CSA2 (compressed sparse attention).

K3 applies no positional encoding here. Position stays in the KDA decay.

Every query attends to two key sets inside one softmax:

* a local band of the last ``csa_local`` raw tokens (keys/values rebuilt
  from the KV latent), and
* compressed entries: every ``csa_group`` consecutive latents are projected
  into one entry. A light indexer scores entries per query and keeps the top
  ``csa_top_k`` among the entries that end before the local band starts. The
  attention logits for the kept entries are the per-head ``q . k_entry``
  products, the same as for raw keys. The indexer only chooses.

The indexer is trained with a KL loss towards the head-summed attention mass
over the kept entries (DeepSeek-V3.2 DSA style). Its inputs are detached, so
this loss does not move the backbone; the layer exposes it as ``_aux_loss``.

Cache. The cache keeps the last raw latents needed for the local band and
for the next incomplete group, plus every compressed entry from position 0.
Entries grow by one ``kv_lora_rank`` vector per ``csa_group`` tokens, so the
cache is O(L / csa_group), not O(1); in exchange cached decoding equals the
no-cache forward at any length. There is no separate attention window.

A full-rank sigmoid gate scales the attention output before the output
projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from train.config import MiniK3Config
from train.models.attention_mask import attention_blocked
from train.models.kv_cache import EntryStore

# Upper bound on the elements of one [batch, heads, queries, keys] score block.
_SCORE_BUDGET = 1 << 24


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
        self._aux_loss = None
        # None = pick the cheaper exact formulation; True/False forces one (tests).
        self._absorb_override = None

    # ------------------------------------------------------------------ helpers
    def _gate(self, hidden_states: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        b, length, _ = hidden_states.shape
        gate = torch.sigmoid(self.out_gate(hidden_states))
        gate = gate.view(b, length, self.num_heads, self.v_head_dim).transpose(1, 2)
        return attn_out * gate.to(attn_out.dtype)

    def _rebuild_kv(self, kv_lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, length, _ = kv_lat.shape
        kv_full = self.kv_up(kv_lat).view(b, length, self.num_heads, self.qk_nope_dim + self.v_head_dim)
        keys = kv_full[..., : self.qk_nope_dim].transpose(1, 2)
        values = kv_full[..., self.qk_nope_dim :].transpose(1, 2)
        return keys, values

    def _project_groups(self, latent: torch.Tensor, first_group: int, last_group: int, origin: int) -> torch.Tensor:
        """Entries for absolute groups [first_group, last_group); ``latent[:, j]`` is position origin+j."""
        group = self.config.csa_group
        batch, _, rank = latent.shape
        count = last_group - first_group
        if count <= 0:
            return latent.new_zeros(batch, 0, rank)
        start = first_group * group - origin
        if start < 0:
            raise RuntimeError("CSA2 cache lost latents of an incomplete group")
        chunk = latent[:, start:start + count * group]
        return self.entry_proj(chunk.reshape(batch, count, group * rank))

    @staticmethod
    def _index_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """Indexer projection in FP32 (or FP64 when the module is FP64)."""
        dtype = linear.weight.dtype if linear.weight.dtype == torch.float64 else torch.float32
        return F.linear(x.to(dtype), linear.weight.to(dtype))

    @staticmethod
    def _cached_entries(past):
        if past is None or past.get("entries") is None:
            return None
        return past["entries"].dense()

    # ---------------------------------------------------------------- attention
    def _attend(self, hidden_states, q, kv_lat, kv_origin, entries, cache_position, memory, mode):
        """q: [B,H,L,D] for positions cache_position.. ; kv_lat[:, j] is position kv_origin + j."""
        batch, heads, length, dim = q.shape
        local = self.config.csa_local
        group = self.config.csa_group
        scale = 1.0 / math.sqrt(dim)
        k_raw, v_raw = self._rebuild_kv(kv_lat)
        complete = 0 if entries is None else entries.shape[1]
        reuse_index = None
        if mode == "reuse" and memory is not None and memory.get("index") is not None:
            reuse_index = memory["index"]
        if complete:
            entry_last = torch.arange(complete, device=q.device) * group + (group - 1)
            n_take = min(self.config.csa_top_k, complete)
            with torch.autocast(device_type=q.device.type, enabled=False):
                index_keys = self._index_linear(self.index_k, entries.detach())
            # Exact alternatives for the entry keys/values:
            #   rebuild: kv_up over every entry once  ~ C*R*H*(Dk+Dv) + L*H*C*(Dk+Dv)
            #   absorb:  fold kv_up into q and into the output ~ L*H*(R*(Dk+Dv) + 2*C*R)
            rank = entries.shape[-1]
            kv_dim = self.qk_nope_dim + self.v_head_dim
            rebuild_cost = complete * rank * kv_dim + length * complete * kv_dim
            absorb_cost = length * (rank * kv_dim + 2 * complete * rank)
            absorb = absorb_cost < rebuild_cost if self._absorb_override is None else self._absorb_override
            if absorb:
                w_kv = self.kv_up.weight.view(heads, kv_dim, rank)
                w_k, w_v = w_kv[:, : self.qk_nope_dim], w_kv[:, self.qk_nope_dim:]
                ent = entries.unsqueeze(1)                                   # [B,1,C,R]
            else:
                k_ent, v_ent = self._rebuild_kv(entries)
        # Band scores are [chunk, chunk + local - 1]; a small block keeps most of them live.
        cap = max(256, 2 * local)
        chunk = max(1, min(length, cap, _SCORE_BUDGET // max(1, batch * heads * (cap + local + complete))))
        outputs, chosen_parts, aux_parts = [], [], []
        for s in range(0, length, chunk):
            e = min(length, s + chunk)
            q_c = q[:, :, s:e]
            q_pos = torch.arange(cache_position + s, cache_position + e, device=q.device)
            # Local band: raw keys in (pos - local, pos].
            k_lo = max(0, cache_position + s - local + 1 - kv_origin)
            k_hi = cache_position + e - kv_origin
            k_pos = torch.arange(kv_origin + k_lo, kv_origin + k_hi, device=q.device)
            band = torch.matmul(q_c, k_raw[:, :, k_lo:k_hi].transpose(-1, -2)).float() * scale
            band = band.masked_fill(attention_blocked(q_pos[:, None], k_pos[None, :], local), float("-inf"))
            scores = band
            if complete:
                with torch.autocast(device_type=q.device.type, enabled=False):
                    index_q = self._index_linear(self.index_q, hidden_states[:, s:e].detach())
                    index_scores = torch.matmul(index_q, index_keys.transpose(-1, -2)) / math.sqrt(index_keys.shape[-1])
                eligible = entry_last.view(1, 1, -1) < (q_pos - local + 1).view(1, -1, 1)
                eligible = eligible.expand(batch, -1, -1)
                if reuse_index is not None:
                    chosen = reuse_index[:, s:e].clamp(0, complete - 1)
                else:
                    masked = index_scores.masked_fill(~eligible, float("-inf"))
                    chosen = torch.topk(masked, k=n_take, dim=-1).indices
                chosen_parts.append(chosen)
                selected = torch.zeros_like(eligible).scatter(2, chosen, True) & eligible
                if absorb:
                    q_abs = torch.einsum("bhld,hdr->bhlr", q_c, w_k.to(q_c.dtype))
                    glob = torch.matmul(q_abs, ent.transpose(-1, -2).to(q_abs.dtype)).float() * scale
                else:
                    glob = torch.matmul(q_c, k_ent.transpose(-1, -2)).float() * scale
                glob = glob.masked_fill(~selected.unsqueeze(1), float("-inf"))
                scores = torch.cat((band, glob), dim=-1)
            weights = torch.softmax(scores, dim=-1)
            w_band = weights[..., : band.shape[-1]].to(v_raw.dtype)
            out = torch.matmul(w_band, v_raw[:, :, k_lo:k_hi])
            if complete:
                w_glob = weights[..., band.shape[-1]:]
                if absorb:
                    mixed_lat = torch.matmul(w_glob.to(ent.dtype), ent)          # [B,H,Lc,R]
                    out = out + torch.einsum("bhlr,hdr->bhld", mixed_lat, w_v.to(mixed_lat.dtype)).to(out.dtype)
                else:
                    out = out + torch.matmul(w_glob.to(v_ent.dtype), v_ent)
                if self.training:
                    aux_parts.append(self._indexer_kl(index_scores, selected, w_glob.detach()))
            outputs.append(out)
        if aux_parts:
            total = sum(part[0] for part in aux_parts)
            rows = sum(part[1] for part in aux_parts)
            self._aux_loss = total / rows.clamp_min(1)
        index = torch.cat(chosen_parts, dim=1) if chosen_parts else None
        self._published = {"entries": entries, "index": index}
        return torch.cat(outputs, dim=2)

    @staticmethod
    def _indexer_kl(index_scores, selected, w_glob):
        """Sum over rows of KL(head-summed attention mass || indexer softmax), kept entries only."""
        target = w_glob.sum(dim=1)                                   # [B, Lc, C]
        target = target * selected
        mass = target.sum(dim=-1, keepdim=True)
        has_rows = mass.squeeze(-1) > 0
        target = target / mass.clamp_min(1e-12)
        logits = index_scores.masked_fill(~selected, torch.finfo(index_scores.dtype).min)
        log_pred = torch.log_softmax(logits, dim=-1)
        kl = (target * (torch.log(target.clamp_min(1e-12)) - log_pred)).masked_fill(~selected, 0.0).sum(-1)
        kl = kl.masked_fill(~has_rows, 0.0)
        return kl.sum(), has_rows.sum()

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv=None, use_cache: bool = False, cache_position: int = 0,
        memory=None, mode: str = "full",
    ):
        del attention_mask
        self._aux_loss = None
        b, l, _ = hidden_states.shape
        group = self.config.csa_group
        q_lat = self.q_norm(self.q_down(hidden_states))
        q = self.q_up(q_lat).view(b, l, self.num_heads, self.qk_nope_dim).transpose(1, 2)
        new_lat = self.kv_norm(self.kv_down(hidden_states))
        if past_kv is not None:
            tail = past_kv["tail"].to(new_lat.dtype)
            kv_lat = torch.cat([tail, new_lat], dim=1)
            kv_origin = cache_position - tail.shape[1]
            old_entries = self._cached_entries(past_kv)
            old_count = int(past_kv["count"])
        else:
            if cache_position != 0:
                raise ValueError("A no-cache MLA forward must start at position 0")
            kv_lat, kv_origin, old_entries, old_count = new_lat, 0, None, 0
        new_count = (cache_position + l) // group
        own_new = None
        if mode == "full":
            own_new = self._project_groups(kv_lat, old_count, new_count, kv_origin)
            if old_entries is not None:
                entries = torch.cat([old_entries.to(own_new.dtype), own_new], dim=1)
            else:
                entries = own_new
            connect = None
        else:
            # Reindex/Reuse read the encoder's entries; this layer's projection
            # is unused by design.  Keep it in the graph with an exact zero.
            entries = None if memory is None else memory.get("entries")
            connect = self.entry_proj.weight.sum() * 0.0
        attn_out = self._attend(hidden_states, q, kv_lat, kv_origin, entries, cache_position, memory, mode)
        if connect is not None:
            attn_out = attn_out + connect.to(attn_out.dtype)
        attn_out = self._gate(hidden_states, attn_out)
        result = self.out_proj(attn_out.transpose(1, 2).contiguous().view(b, l, -1))
        if not use_cache:
            return result
        return result, self._store(past_kv, kv_lat, own_new, new_count)

    def _store(self, past_kv, kv_lat, own_new, new_count):
        keep = max(self.config.csa_local, self.config.csa_group)
        store = None if past_kv is None else past_kv.get("entries")
        if own_new is not None and own_new.shape[1]:
            if store is None:
                store = EntryStore(fp4=bool(self.config.kv_cache_fp4))
            store.append(own_new.detach())
        return {
            "tail": kv_lat[:, -keep:].detach(),
            "entries": store,
            "count": new_count,
        }
