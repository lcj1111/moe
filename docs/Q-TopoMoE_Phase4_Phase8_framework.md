# Phase 4 / Phase 8 executable framework

This commit adds the CPU-side control plane. It does not claim a CUDA
benchmark: a kernel row is selectable by default only when `measured: true`,
`valid: true`, and it contains a p50 or p95 latency.  The empty kernel DB
template is therefore safe to check in before CUTLASS/Triton/FlashInfer
measurements are collected.

## M-bucket workload generation

```bash
python3 phase4/workload/generate_m_buckets.py \
  --output configs/workloads/m_buckets.json --seed 42
```

The manifest has 90 deterministic cases: W1 (256/128, C=1/8/32), W2
(2048/256, C=1/8/32), W3 (8192/256, C=1/8/16), W4 (32768/128, C=1), crossed
with prefix-cache 0/50/100% and closed-loop/Poisson/burst arrivals. `prefill_m`
is `input_tokens * concurrency`; `decode_m` is `concurrency`. Both are
rounded up to a configured power-of-two bucket. `real_M_hist` keeps the
prefill/decode mixture used by Phase 8's cost model.

## Phase 4 kernel/backend selector

Populate `configs/kernels/phase4_kernel_db.json` with measured rows, for
example `backend=cutlass`, `kernel_config={tile_m,tile_n,tile_k,stages,warps}`,
`m_bucket`, `precision`, `p50_us`, `p95_us`, `measured=true`, and a source run
ID. Then load `KernelDatabase` and call `BackendSelector.select`. Missing or
invalid rows raise `SelectionError`; `allow_unmeasured=True` is an explicit
development-only escape hatch.

## Phase 8 strategy selector

`StrategyCandidate` contains the Runbook fields: quant format/checkpoint,
TP/DP/EP, GPU mapping, EPLB policy, redundant experts, kernel backend and
kernel config. `StrategySelector` first eliminates unsupported, quality-invalid
and over-memory candidates. `CostModel` then evaluates:

`compute(real_M_hist, kernel_db) + communication_bytes * measured mapping cost +
imbalance(route_hist) + migration_bytes * migration cost`. Supply measured
mapping rates through `CostModel.communication_us_per_gb_by_mapping`; the
default rate is only a documented fallback for a pre-measurement dry run.

`evaluate()` reports top-1 choice, median/p95 regret, decision overhead as a
percentage of predicted p99, and invalid configuration rate. The Runbook gates
(median regret <=5%, p95 <=10%, controller overhead <1%) are output as explicit
fields; no RL is introduced until the transparent model fails those gates.

## CPU verification

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
python3 -m py_compile phase4/workload/generate_m_buckets.py selector/*.py scripts/analyze_phase1.py
```

No GPU process is started by these commands.
