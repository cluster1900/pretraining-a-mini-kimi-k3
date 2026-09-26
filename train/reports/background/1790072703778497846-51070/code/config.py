"""
Unified single-model configuration for Mini Kimi K3 (1.02B Total / 145M Active).
Tailored for training on 4 Tesla V100-SXM2 GPUs (32GB each).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MiniK3Config:
    """
    Mini Kimi K3 Canonical Configuration (1.02B total parameters, 145M active parameters).
    
    Structure:
    - 12 layers total: 9 KDA (Kimi Delta Attention) + 3 MLA (Multi-Head Latent Attention)
    - 3:1 KDA to MLA ratio, full attention on layers 4, 8, 12
    - MoE: 256 routed experts (top-6) + 2 shared experts (always active)
    - Latent MoE projection: hidden // 2 = 256
    - situ activation: beta=4.0, linear_beta=25.0
    - Vocab: 163,840 (official K3 BPE) with tied embeddings
    """
    model_name: str = "mini-k3-1.02b"
    
    # Dimensions
    hidden_size: int = 512
    num_layers: int = 12
    vocab_size: int = 163840
    max_position_embeddings: int = 1_048_576
    attention_window: int = 4096
    rope_theta: float = 10_000_000.0          # Base frequency for 1M context RoPE
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Attention Stacking (3:1 ratio)
    head_dim: int = 128
    num_attention_heads: int = 8       # MLA heads (8 * 128 = 1024)
    num_kda_heads: int = 4             # KDA heads (4 * 128 = 512 == hidden_size)
    mla_layers: List[int] = field(default_factory=lambda: [4, 8, 12])
    
    # MLA specifics (nope: 128, rope: 64, value: 128)
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
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
    
    # Router & Balancer
    topk_method: str = "noaux_tc"
    balancer_gamma: float = 1e-2             # Optimal gamma measured for batch size 131k
    
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
        if self.top_k > self.num_routed_experts or self.top_k < 1:
            raise ValueError("top_k must be in [1, num_routed_experts]")
        if self.micro_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch sizes must be positive")
        if abs(sum(self.stable_mix.values()) - 1.0) > 1e-6:
            raise ValueError("stable_mix must sum to 1")
        if abs(sum(self.decay_mix.values()) - 1.0) > 1e-6:
            raise ValueError("decay_mix must sum to 1")


# Default canonical config instance
DEFAULT_CONFIG = MiniK3Config()
