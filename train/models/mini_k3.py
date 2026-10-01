"""
Mini Kimi K3 Causal Language Model.
Assembles:
- 13 decoder layers: 9 KDA (Linear Attention) + 4 MLA (gated latent attention, Layers 4, 8, 12, 13)
- Layer 0 Dense MLP + Layers 1..11 Latent MoE (256 routed experts + 2 shared experts)
- RMSNorm pre-layer normalizations
- Weight-tied word embeddings and LM head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from train.config import MiniK3Config, DEFAULT_CONFIG
from train.models.kda import KimiDeltaAttention
from train.models.mla import MultiHeadLatentAttention
from train.models.moe import KimiMoEBlock, ExpertMLP
from train.engine.init_patch import apply_init_patches
from train.models.kv_cache import KVCache
from train.models.mtp import MTPBlock
from train.models.attn_res import AttentionResidual
from train.models.mhc import MHCCoeffs, collapse_streams
from train.models.engram import Engram
from train.models.moonvit import MoonViT


class DenseMLP(nn.Module):
    """Layer 0 Dense MLP following first_k_dense_replace: 1."""
    def __init__(self, config: MiniK3Config):
        super().__init__()
        # Standard intermediate dimension for dense layer
        intermediate = config.hidden_size * 4
        self.mlp = ExpertMLP(
            in_features=config.hidden_size,
            intermediate_features=intermediate,
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MiniK3DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: MiniK3Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # 1. Pre-norm for Attention
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Layer 4, 8, 12 (1-indexed, i.e., idx 3, 7, 11) use MLA; others use KDA
        is_mla = (layer_idx + 1) in config.mla_layers
        if is_mla:
            self.self_attn = MultiHeadLatentAttention(config)
            self.is_mla = True
        else:
            self.self_attn = KimiDeltaAttention(config)
            self.is_mla = False

        # 2. Pre-norm for MLP / MoE
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
            self.is_moe = False
        else:
            self.mlp = KimiMoEBlock(config)
            self.is_moe = True
        self.mhc_attn = MHCCoeffs(config.hidden_size, config.mhc_streams, config.mhc_sinkhorn_iters)
        self.mhc_mlp = MHCCoeffs(config.hidden_size, config.mhc_streams, config.mhc_sinkhorn_iters)

    def _run(self, streams, pre, attn_input, state, conv, past, use_cache, cache_position, memory, mode):
        if self.is_mla:
            result = self.self_attn(
                self.input_layernorm(attn_input), past_kv=past, use_cache=use_cache,
                cache_position=cache_position, memory=memory, mode=mode,
            )
            attn_out, cache = result if use_cache else (result, None)
            new_state, new_conv = None, None
        else:
            attn_out, new_state, new_conv = self.self_attn(
                self.input_layernorm(attn_input), recurrent_state=state, conv_state=conv,
            )
            cache = None
        streams, pre = self.mhc_attn.mix(streams, attn_out, attn_input, previous_pre=pre)
        hidden = collapse_streams(streams, pre)
        mlp_out = self.mlp(self.post_attention_layernorm(hidden))
        streams, pre = self.mhc_mlp.mix(streams, mlp_out, hidden, previous_pre=pre)
        return streams, pre, new_state, new_conv, cache

    def forward(self, streams, pre, attn_input, state=None, conv=None, past=None,
                use_cache=False, cache_position=0, memory=None, mode="full"):
        entries = None if memory is None else memory.get("entries")
        index = None if memory is None else memory.get("index")
        use_ckpt = self.training and self.config.activation_checkpointing and not use_cache
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
            empty_entries = attn_input.new_zeros(0)
            empty_index = torch.zeros(0, dtype=torch.long, device=attn_input.device)

            def train_only(stream_in, pre_in, attn_in, entry_in, index_in):
                mem = None
                if entry_in.numel() > 0 and mode != "full":
                    mem = {"entries": entry_in, "index": None if index_in.numel() == 0 else index_in}
                next_streams, next_pre, _, _, _ = self._run(
                    stream_in, pre_in, attn_in, None, None, None, False, 0, mem, mode,
                )
                return next_streams, next_pre

            streams, pre = checkpoint(
                train_only, streams, pre, attn_input,
                entries if entries is not None else empty_entries,
                index if index is not None else empty_index,
                use_reentrant=False,
            )
            new_state, new_conv, cache = None, None, None
        else:
            streams, pre, new_state, new_conv, cache = self._run(
                streams, pre, attn_input, state, conv, past, use_cache, cache_position, memory, mode,
            )
        if self.training and self.is_moe and self.mlp.gate._last_counts is not None:
            self.mlp.gate.expert_load.add_(self.mlp.gate._last_counts)
            self.mlp.gate.accumulate_margin_histogram()
        return streams, pre, new_state, new_conv, cache


class MiniK3ForCausalLM(nn.Module):
    def __init__(self, config: Optional[MiniK3Config] = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size

        # Token embedding
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Decoder layers. The default stack is 9 KDA + 4 MLA.
        self.layers = nn.ModuleList([
            MiniK3DecoderLayer(i, self.config) for i in range(self.config.num_layers)
        ])

        self.layer_mix = nn.ModuleList([
            AttentionResidual(self.hidden_size, self.config.rms_norm_eps)
            for _ in range(self.config.num_layers)
        ])
        self.final_mix = AttentionResidual(self.hidden_size, self.config.rms_norm_eps)
        self.mhc_pre = nn.Parameter(torch.zeros(self.config.mhc_streams))
        self.mhc_pre.data[0] = 8
        self.engrams = nn.ModuleDict({
            str(layer_id): Engram(self.config, layer_id)
            for layer_id in self.config.engram_layers
            if 1 <= layer_id <= self.config.num_layers
        })
        self.vision = MoonViT(self.config)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)
        self.mtp = MTPBlock(self.config) if getattr(self.config, "mtp_enabled", False) else None
        self.mtp_lambda = float(getattr(self.config, "mtp_lambda", 0.3))

        # Output LM Head (tied with embed_tokens)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.embed_tokens.weight.adam_only = True

        # Apply robust initialization patches
        apply_init_patches(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def _sample(self, logits, temperature, top_p):
        logits = logits.float()
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        probs = torch.softmax(logits, -1)
        if top_p < 1.0:
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            # Retain the first token that crosses the nucleus threshold.
            keep = (torch.cumsum(sorted_p, -1) - sorted_p) < top_p
            keep[..., 0] = True
            probs = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
            probs = probs / probs.sum(-1, keepdim=True)
            return sorted_i.gather(-1, torch.multinomial(probs, 1))
        return torch.multinomial(probs, 1)

    def _verify_draft(self, target_logits, draft_logits, draft_id, temperature):
        """Accept the MTP draft when it agrees with the main head; otherwise resample."""
        if temperature <= 0:
            target_id = target_logits.argmax(-1, keepdim=True)
            return bool((target_id == draft_id).all()), target_id
        target = torch.softmax(target_logits.float() / temperature, dim=-1)
        draft = torch.softmax(draft_logits.float() / temperature, dim=-1)
        draft_prob = draft.gather(-1, draft_id).clamp_min(1e-8)
        target_prob = target.gather(-1, draft_id)
        accept = torch.log(torch.rand_like(target_prob)) < (target_prob / draft_prob).clamp(max=1).log()
        if bool(accept.all()):
            return True, draft_id
        residual = (target - draft).clamp_min(0)
        residual = residual / residual.sum(-1, keepdim=True).clamp_min(1e-8)
        return False, torch.multinomial(residual, 1)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 1.0, top_p: float = 1.0,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Sample the target model using its cache; MTP remains a training loss.

        The former speculative path verified unfiltered probabilities after
        sampling from top-p and was not distribution preserving.
        """
        if max_new_tokens < 0 or not 0 < top_p <= 1:
            raise ValueError("max_new_tokens must be nonnegative and top_p in (0, 1]")
        if max_new_tokens == 0:
            return input_ids
        self.eval()
        cache = self.new_kv_cache()
        out = self(input_ids, use_cache=True, past_key_values=cache)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            next_id = self._sample(out["logits"][:, -1], temperature, top_p)
            if eos_token_id is not None:
                next_id = torch.where(finished[:, None], eos_token_id, next_id)
                finished |= next_id.squeeze(-1).eq(eos_token_id)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if finished.all() or step + 1 == max_new_tokens:
                break
            out = self(next_id, use_cache=True, past_key_values=cache)
        return input_ids

    def new_kv_cache(self):
        """Create an empty cache with one slot per decoder layer."""
        return KVCache(
            kda_states=[None] * self.config.num_layers,
            kda_conv_states=[None] * self.config.num_layers,
            mla_keys=[None] * self.config.num_layers,
            mla_values=[None] * self.config.num_layers,
        )

    def _mla_mode(self, layer_index: int) -> str:
        if layer_index < self.config.encoder_layers:
            return "full"
        decoder = [
            index for index in range(self.config.num_layers)
            if (index + 1) in self.config.mla_layers and index >= self.config.encoder_layers
        ]
        if layer_index not in decoder:
            return "full"
        return "reindex" if decoder.index(layer_index) % 2 == 0 else "reuse"

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None, use_cache: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        return_hidden: bool = False, compute_logits: bool = True,
    ) -> Dict[str, Any]:
        """
        input_ids: [B, L]
        labels: [B, L] (optional)
        pixel_values: [B, 3, H, W] or [B, T, 3, H, W]
        """
        if labels is not None and not compute_logits:
            raise ValueError("labels require compute_logits=True")
        if use_cache and past_key_values is None:
            past_key_values = self.new_kv_cache()
        embedded = self.embed_tokens(input_ids)
        text_offset = 0
        if pixel_values is not None:
            visual = self.vision(pixel_values)
            text_offset = visual.shape[1]
            embedded = torch.cat((visual, embedded), dim=1)
            if labels is not None:
                pad = labels.new_full((labels.shape[0], text_offset), -100)
                labels = torch.cat((pad, labels), dim=1)
        batch, length, _ = embedded.shape
        streams = embedded.new_zeros(batch, length, self.config.mhc_streams, self.hidden_size)
        streams[:, :, 0] = embedded
        pre = torch.softmax(self.mhc_pre.float(), dim=0).to(dtype=embedded.dtype)
        pre = pre.view(1, 1, -1).expand(batch, length, -1)
        sources = [embedded]
        encoder_memory = None
        previous_memory = None
        position = past_key_values.position if past_key_values is not None else 0
        if past_key_values is not None and past_key_values.token_ids is not None:
            text_ids = torch.cat((past_key_values.token_ids, input_ids), dim=1)
        else:
            text_ids = input_ids

        for i, layer in enumerate(self.layers):
            mode = self._mla_mode(i) if layer.is_mla else "full"
            memory = None
            if layer.is_mla and mode == "reindex":
                memory = encoder_memory
            elif layer.is_mla and mode == "reuse":
                memory = previous_memory
            past = past_key_values.mla_keys[i] if past_key_values is not None and layer.is_mla else None
            state = past_key_values.kda_states[i] if past_key_values is not None else None
            conv = past_key_values.kda_conv_states[i] if past_key_values is not None else None
            streams, pre, state, conv, cache = layer(
                streams, pre, self.layer_mix[i](sources), state=state, conv=conv, past=past,
                use_cache=use_cache, cache_position=position, memory=memory, mode=mode,
            )
            collapsed = collapse_streams(streams, pre)
            engram = self.engrams[str(i + 1)] if str(i + 1) in self.engrams else None
            if engram is not None:
                delta = engram(text_ids, collapsed[:, text_offset:])
                collapsed = collapsed.clone()
                collapsed[:, text_offset:] = collapsed[:, text_offset:] + delta
                streams = streams.clone()
                streams[:, text_offset:, 0] = streams[:, text_offset:, 0] + delta
            sources.append(collapsed)
            if use_cache and past_key_values is not None:
                if layer.is_mla:
                    past_key_values.mla_keys[i] = cache
                else:
                    past_key_values.kda_states[i] = None if state is None else state.detach()
                    past_key_values.kda_conv_states[i] = conv
            if layer.is_mla:
                published = layer.self_attn._published
                if i < self.config.encoder_layers:
                    encoder_memory = published
                previous_memory = published

        hidden_states = self.norm(self.final_mix(sources))
        token_embed = embedded
        logits = self.lm_head(hidden_states) if compute_logits else None

        loss = None
        lm_loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss
            if self.mtp is not None and labels.size(1) >= 3 and (labels[:, 2:] != -100).any():
                mtp_hidden = self.mtp(hidden_states, token_embed)
                mtp_logits = self.lm_head(mtp_hidden)
                mtp_labels = labels[:, 2:]
                mtp_loss = F.cross_entropy(
                    mtp_logits.reshape(-1, self.vocab_size),
                    mtp_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = lm_loss + self.mtp_lambda * mtp_loss

        if use_cache and past_key_values is not None:
            past_key_values.position += length
            # N-gram history plus the three preceding convolution positions.
            history = self.config.engram_max_ngram - 1 + 3
            past_key_values.token_ids = text_ids[:, -history:].detach()
        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "hidden": hidden_states if use_cache or return_hidden else None,
            "logits": logits, "past_key_values": past_key_values if use_cache else None,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Counts total, active, and non-embedding active parameters."""
        total = sum(p.numel() for p in self.parameters())
        expert_params = 0
        engram_table = 0
        vision = 0
        for name, p in self.named_parameters():
            if ".experts." in name and "shared_experts" not in name:
                expert_params += p.numel()
            elif ".tables." in name:
                engram_table += p.numel()
            elif name.startswith("vision."):
                vision += p.numel()
        routed_active = int(expert_params * self.config.top_k / self.config.num_routed_experts)
        engram_active = 0
        if self.engrams:
            slots = (self.config.engram_max_ngram - 1) * self.config.engram_heads
            engram_active = len(self.engrams) * slots * self.config.engram_head_dim
        active = total - expert_params - engram_table - vision + routed_active + engram_active
        embed_params = self.embed_tokens.weight.numel()
        return {
            "total": total,
            "active": active,
            "embed": embed_params,
            "non_embed_active": active - embed_params,
            "routed_expert_params": expert_params,
            "engram_table": engram_table,
            "vision": vision,
        }
