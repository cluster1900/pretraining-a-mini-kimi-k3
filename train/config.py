"""
Unified single-model configuration for Mini Kimi K3 (1,150,739,900 total / 159,400,252 active).
Tailored for training on 4 Tesla V100-SXM2 GPUs (32GB each).
"""

from dataclasses import dataclass, field
from typing import List


# The parameter contract is part of the audited run identity. A smoke report
# from the older 12-layer prototype must never authorize the 13-layer run.
CANONICAL_PARAMETER_COUNTS = {
    "mini-k3": {"total": 1_150_739_900, "active": 159_400_252},
}


@dataclass
class MiniK3Config:
    """
    Mini Kimi K3 Canonical Configuration (1,150,739,900 total parameters, 159,400,252 active).

    Structure:
    - 13 layers: 9 KDA + 4 gated NoPE MLA, MLA on layers 4, 8, 12, 13
    - Encoder layers 1-8, decoder layers 9-13, CSA2 on the MLA layers
    - 4-stream single-pass mHC, Engram at layers 2 and 8, scaled MoonViT-V2
    - MoE: 256 routed experts (top-6) + 2 shared experts
    - Vocab: 163,840 with tied embeddings
    """
    model_name: str = "mini-k3"
    
    # Dimensions
    hidden_size: int = 512
    num_layers: int = 13
    vocab_size: int = 163840
    max_position_embeddings: int = 1_048_576
    attention_window: int = 4096
    rope_theta: float = 10_000_000.0          # Unused. MLA is NoPE; position is carried by KDA.
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Attention Stacking (3:1 ratio)
    head_dim: int = 128
    num_attention_heads: int = 8       # MLA heads (8 * 128 = 1024)
    num_kda_heads: int = 4             # KDA heads (4 * 128 = 512 == hidden_size)
    mla_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 13])
    encoder_layers: int = 8
    
    # MLA is NoPE. qk_rope_head_dim stays at 0 so older callers can still read it.
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 0
    v_head_dim: int = 128
    kv_lora_rank: int = 256
    q_lora_rank: int = 512
    
    # KDA specifics
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    
    # MoE Architecture
    num_routed_experts: int = 256
    num_shared_experts: int = 2
    top_k: int = 6
    moe_intermediate_size: int = 416
    routed_expert_hidden_size: int = 256     # hidden_size // 2
    first_k_dense_replace: int = 1           # Layer 0 is dense MLP
    
    # Router & Balancer. Quantile balancing ignores balancer_gamma.
    topk_method: str = "quantile"
    balancer_gamma: float = 1e-2
    
    # situ activation parameters
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    
    # Training Parameters on 4x V100-SXM2 (32GB each). V100 has no BF16
    # Tensor Cores, so the runner must use FP16 GradScaler.
    micro_batch_size: int = 1                # 1 seq * 2048 tokens per forward
    gradient_accumulation_steps: int = 32    # 4 GPUs * 1 * 32 * 2048 = 262,144 tokens/step
    sequence_length: int = 2048
    precision: str = "fp16"
    distributed: bool = True
    activation_checkpointing: bool = True
    mtp_enabled: bool = True
    mtp_lambda: float = 0.3
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    csa_group: int = 4
    csa_top_k: int = 512
    csa_local: int = 128
    csa_index_dim: int = 64
    # FP4 cache is an opt-in memory experiment.  The default must preserve
    # the logits/cache equivalence gate used before any long-context claim.
    kv_cache_fp4: bool = False
    engram_layers: List[int] = field(default_factory=lambda: [2, 8])
    engram_max_ngram: int = 4
    engram_heads: int = 8
    engram_head_dim: int = 32
    engram_table_size: int = 10007
    vision_layers: int = 4
    vision_hidden: int = 512
    vision_heads: int = 8
    vision_patch: int = 14
    data_root: str = "/data/mini-k3/data"
    checkpoint_root: str = "/data/mini-k3/checkpoints"
    log_root: str = "/data/mini-k3/logs"
    rollout_root: str = "/data/mini-k3/rollouts"
    total_steps: int = 38147                 # 10,000,007,168 total tokens on 4x V100 (262,144 tokens/step)
    peak_lr: float = 6.00e-4
    warmup_frac: float = 0.02                # ~762 steps
    decay_frac: float = 0.15                 # Starts at step 32,430
    min_lr_frac: float = 0.10                # Linear decay down to 6.00e-5
    weight_decay: float = 0.10
    grad_clip: float = 1.0

    # Data curriculum.  These are deliberately explicit so the decay phase
    # cannot silently continue using the stable mixture.
    stable_mix: dict = field(default_factory=lambda: {
        "fineweb-edu": 0.45, "chinese-fineweb-edu": 0.15, "dolma-body": 0.10,
        "finemath": 0.05, "open-web-math": 0.05, "code-python": 0.10,
        "cosmopedia": 0.10,
    })
    decay_mix: dict = field(default_factory=lambda: {
        "fineweb-edu": 0.30, "chinese-fineweb-edu": 0.10, "dolma-body": 0.10,
        "finemath": 0.12, "open-web-math": 0.08, "code-python": 0.25,
        "cosmopedia": 0.05,
    })

    def validate(self) -> None:
        if self.top_k >= self.num_routed_experts or self.top_k < 1:
            raise ValueError("top_k must leave one rejected expert for quantile balancing")
        if self.gate_lower_bound >= 0:
            raise ValueError("gate_lower_bound is the negative KDA log-decay floor")
        if self.num_layers < 1 or self.mhc_streams < 2 or self.mhc_sinkhorn_iters < 1:
            raise ValueError("layers and mHC streams must be positive")
        if self.csa_group < 1 or self.csa_local < 1 or self.csa_top_k < 1:
            raise ValueError("CSA2 group, local window, and top-k must be positive")
        if any(layer < 1 or layer > self.num_layers for layer in self.mla_layers):
            raise ValueError("mla_layers must point at real layers")
        if self.encoder_layers < 0:
            raise ValueError("encoder_layers must be non-negative")
        if self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch sizes must be positive")
        if abs(sum(self.stable_mix.values()) - 1.0) > 1e-6:
            raise ValueError("stable_mix must sum to 1")
        if abs(sum(self.decay_mix.values()) - 1.0) > 1e-6:
            raise ValueError("decay_mix must sum to 1")


# Default canonical config instance
DEFAULT_CONFIG = MiniK3Config()
