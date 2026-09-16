"""
Mini Kimi K3 Causal Language Model.
Assembles:
- 12 decoder layers: 9 KDA (Linear Attention) + 3 MLA (Full Attention, Layers 4, 8, 12)
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
        
        # Layer 0 is Dense MLP; Layer 1..11 are MoE blocks
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
            self.is_moe = False
        else:
            self.mlp = KimiMoEBlock(config)
            self.is_moe = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
        conv_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        # Residual 1: Attention block
        normed = self.input_layernorm(hidden_states)
        if self.is_mla:
            attn_out = self.self_attn(normed)
            new_state = None
            new_conv = None
        else:
            attn_out, new_state, new_conv = self.self_attn(normed, recurrent_state=recurrent_state, conv_state=conv_state)
        hidden_states = hidden_states + attn_out

        # Residual 2: MLP / MoE block
        normed_mlp = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp)
        hidden_states = hidden_states + mlp_out

        return hidden_states, new_state, new_conv


class MiniK3ForCausalLM(nn.Module):
    def __init__(self, config: Optional[MiniK3Config] = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size

        # Token embedding
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # 12 Decoder layers
        self.layers = nn.ModuleList([
            MiniK3DecoderLayer(i, self.config) for i in range(self.config.num_layers)
        ])

        # Final normalization
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)

        # Output LM Head (tied with embed_tokens)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Apply robust initialization patches
        apply_init_patches(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 1.0, top_p: float = 1.0,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Small, deterministic-compatible sampler for SFT/RL rollouts.

        This reference implementation recomputes the prefix each step; it is
        intentionally correct and easy to audit. Production rollouts should
        replace it with a KV-cache engine after validating token equivalence.
        """
        self.eval(); cache = self.new_kv_cache()
        # Prime all layer caches with the prompt, then decode one token at a time.
        out = self(input_ids, use_cache=True, past_key_values=cache)
        eos = eos_token_id
        for _ in range(max_new_tokens):
            logits = out["logits"][:, -1].float()
            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, -1)
                if top_p < 1.0:
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    keep = torch.cumsum(sorted_p, -1) <= top_p
                    keep[..., 0] = True
                    probs = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
                    probs = probs / probs.sum(-1, keepdim=True)
                    next_id = sorted_i.gather(-1, torch.multinomial(probs, 1))
                else:
                    next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos is not None and bool((next_id == eos).all()):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None, use_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        input_ids: [B, L]
        labels: [B, L] (optional)
        """
        hidden_states = self.embed_tokens(input_ids)

        presents = []
        for i, layer in enumerate(self.layers):
            if past_key_values is not None:
                past = (past_key_values.mla_keys[i], past_key_values.mla_values[i]) if layer.is_mla and past_key_values.mla_keys[i] is not None else None
                state = past_key_values.kda_states[i]
                conv = past_key_values.kda_conv_states[i]
            else:
                past = None
                state = None
                conv = None
            if layer.is_mla:
                h = layer.input_layernorm(hidden_states)
                attn = layer.self_attn(h, past_kv=past, use_cache=use_cache, cache_position=(past_key_values.position if past_key_values else 0))
                if use_cache: attn, kv = attn; presents.append((None, kv))
                hidden_states = hidden_states + attn
                hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
                if use_cache and past_key_values is not None: past_key_values.mla_keys[i], past_key_values.mla_values[i] = kv
            else:
                hidden_states, state, conv = layer(hidden_states, recurrent_state=state, conv_state=conv)
                if use_cache and past_key_values is not None:
                    past_key_values.kda_states[i] = state.detach()
                    past_key_values.kda_conv_states[i] = conv

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if use_cache and past_key_values is not None: past_key_values.position += input_ids.size(1)
        return {
            "loss": loss,
            "logits": logits, "past_key_values": past_key_values if use_cache else None,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Counts total, active, and non-embedding active parameters."""
        total = sum(p.numel() for p in self.parameters())
        
        # Count routed expert parameters
        expert_params = 0
        for name, p in self.named_parameters():
            if ".experts." in name:
                expert_params += p.numel()
                
        # Active params: always-on params + (top_k / n_experts) * routed_experts
        active = total - expert_params + int(expert_params * self.config.top_k / self.config.num_routed_experts)
        embed_params = self.embed_tokens.weight.numel()
        non_embed_active = active - embed_params

        return {
            "total": total,
            "active": active,
            "embed": embed_params,
            "non_embed_active": non_embed_active,
            "routed_expert_params": expert_params,
        }
