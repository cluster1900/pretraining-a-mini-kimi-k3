# 2026-10-01 v100 verification

These records cover the current `prepared-v2-supplement-v2` manifest and the
13-layer `mini-k3` code. They are not a training checkpoint and do not certify
2048-token throughput or 1M long-context quality.

- Remote host: `v100`, Tesla V100-SXM2-32GB.
- Training manifest: `d2cce19c492c5204bfd369578581ecf5a60f772087da45f002d516f8a2e1b7ea`.
- Validation manifest: `541136c5a7f06cd6606513cecddc2bf20af8ace6cca5cb5ad92df503ded029c7`.
- `AUDIT.json`: `9b99af75062aa0645a8de0fdf9d1a5f06d325ed7118ff1638341071c1d7f2632`.
- Fixed-mix coverage: `sufficient_fixed_mix`; target `10,000,007,168` tokens; available pretraining pool `28,214,510,016` tokens.
- Readiness command returned `status: ready`.
- The stale controller failure was reconciled atomically; the old status hash is recorded in the new `PIPELINE_STATUS.json`.

The full-model smoke used all seven pretraining sources, sequence length 64,
and two optimizer steps on one V100. It passed with the expected parameter
signature (`1,150,739,900` total, `159,400,252` active), finite losses and
gradients, and peak allocation `15,136,763,904` bytes. This is a functional
check only.

The small-model cache check passed with maximum logits difference `4.62e-6`
and identical greedy tokens. The memory check passed its small live model and
theoretical bounded-window checks. The long-document benchmark has not passed
because no trained checkpoint exists yet.

The follow-up contract checks also passed locally and on `v100`: a 2048
short-run budget is accepted when the audited coverage is larger than that
budget, longer training sequences require an explicit 2048 checkpoint and
lower continuation learning rate, and read-only validation evaluation checks
the validation manifest's audit/hash/tokenizer bindings first. No pretraining,
download, or service restart was performed.
