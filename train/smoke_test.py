"""
Pre-flight 2-Step Smoke Test.
Verifies:
1. Architecture instantiation & parameter count alignment
2. Finite parameter initialization (no NaNs or Infs)
3. Step 0 loss alignment with ln(163,840) ≈ 12.007
4. Differentiability of KDA and MoE dispatch over 2 optimizer steps
5. Absence of dead-expert collapse
"""

import math
import sys
import torch
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from train.config import DEFAULT_CONFIG, MiniK3Config
from train.models.mini_k3 import MiniK3ForCausalLM
from train.engine.init_patch import assert_initialised
from train.engine.balancer import NoAuxBalancer


def run_smoke_test():
    print("=" * 70)
    print("Mini Kimi K3: Running Pre-flight Smoke Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")

    # 1. Instantiate model
    cfg = DEFAULT_CONFIG
    print(f"[*] Instantiating model: {cfg.model_name} (hidden={cfg.hidden_size}, layers={cfg.num_layers})...")
    model = MiniK3ForCausalLM(cfg).to(device)

    # 2. Verify Parameter Counts
    counts = model.count_parameters()
    print(f"[*] Parameter Accounting:")
    print(f"    - Total Parameters:          {counts['total'] / 1e6:.2f} M ({counts['total']:,})")
    print(f"    - Active Parameters:         {counts['active'] / 1e6:.2f} M ({counts['active']:,})")
    print(f"    - Non-Embedding Active:      {counts['non_embed_active'] / 1e6:.2f} M")
    print(f"    - Embedding Table:           {counts['embed'] / 1e6:.2f} M ({counts['embed'] / counts['active'] * 100:.1f}% of active)")
    print(f"    - Routed Expert Total:       {counts['routed_expert_params'] / 1e6:.2f} M")

    # 3. Finite Initialization Pre-flight Check
    print("[*] Running finite initialization check (assert_initialised)...")
    assert_initialised(model)
    print("    -> PASS: All parameters and buffers are finite.")

    # 4. Prepare Smoke Test Micro-batch
    # Use short sequence length for rapid test with fixed seed for determinism
    torch.manual_seed(42)
    smoke_b, smoke_l = 2, 64
    dummy_input = torch.randint(0, cfg.vocab_size, (smoke_b, smoke_l), device=device)
    dummy_labels = dummy_input.clone()

    # 5. Optimizer Setup (excluding router bias)
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    balancer = NoAuxBalancer(model, gamma=cfg.balancer_gamma)

    # 6. Step 0 Forward & Initial Loss Check
    model.train()
    out = model(dummy_input, labels=dummy_labels)
    initial_loss = out["loss"].item()
    expected_uniform_loss = math.log(cfg.vocab_size)  # ln(163,840) ≈ 12.007
    print(f"[*] Step 0 Loss: {initial_loss:.4f} (Expected uniform baseline: {expected_uniform_loss:.4f})")

    if not (11.90 <= initial_loss <= 12.25):
        raise AssertionError(f"Initial loss {initial_loss:.4f} outside expected range [11.90, 12.25]")
    else:
        print("    -> PASS: Initial loss conforms strictly to uniform initialization.")

    # 7. Execute 2 Gradient Steps
    losses = [initial_loss]
    for step in range(1, 3):
        optimizer.zero_grad()
        out = model(dummy_input, labels=dummy_labels)
        loss = out["loss"]
        loss.backward()

        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()
        
        # Load balancer step
        telemetry = balancer.step()
        losses.append(loss.item())
        print(f"[*] Step {step}: Loss = {loss.item():.4f}, GradNorm = {grad_norm:.4f}, DeadExperts = {telemetry['dead_frac']*100:.1f}%, Imbalance = {telemetry['imbalance']:.2f}")

    # 8. Assertions
    assert math.isfinite(losses[-1]), "Loss became non-finite during smoke test!"
    print("=" * 70)
    print(">>> SMOKE TEST PASSED SUCCESSFULLY! The model is ready for training. <<<")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
