# Phase 4 / Phase 8 executable framework

This commit adds the CPU-side control plane. It does not claim a CUDA
benchmark: a kernel row is selectable by default only when `measured: true`,
`valid: true`, and it contains a p50 or p95 latency.  The empty kernel DB
template is therefore safe to check in before CUTLASS/Triton/FlashInfer
measurements are collected.

The official RedHatAI NVFP4 geometry now has native
`vllm.run_cutlass_moe_fp4` rows for the accepted trace buckets.  Fifty-repeat
p95 latencies are M=1: 250.42 us, M=2048: 728.23 us, and M=8192: 2219.58 us;
the raw rows are in
`docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_real_m_20260809.json`.  These direct
CUTLASS measurements must not be substituted for EP serving, where the
frozen vLLM build selected MARLIN after expert sharding.

The Runbook screen required seven additional NVFP4 buckets. On the same
frozen RTX 5090/CUTLASS path, 50-repeat p50/p95 values in microseconds are:
M=4 235.61/253.71, M=8 232.72/240.26, M=16 285.39/293.39, M=32
361.28/369.05, M=128 453.48/461.00, M=256 475.03/485.84, and M=16384
4382.33/4413.00. Together with M=1/2048/8192, these cover every prefill and
decode bucket in the frozen 12-cell workload matrix. Raw rows are in
`docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_phase8_missing_m_20260809.json`.

## M-bucket workload generation

```bash
python3 phase4/workload/generate_m_buckets.py \
  --output configs/workloads/m_buckets.json --seed 42
```

The manifest has 108 deterministic cases: W1 (256/128, C=1/8/32/128), W2
(2048/256, C=1/8/32), W3 (8192/256, C=1/8/16), W4 (32768/128, C=1/4), crossed
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

## NVFP4 single-pass screen and selector status (2026-08-09)

Official RedHatAI NVFP4 static EP4 and static EP8 completed the exact Runbook
matrix: W1 256/128 at C=1/8/32/128, W2 2048/256 at C=1/8/32, W3 8192/256 at
C=1/8/16, and W4 32768/128 at C=1/4. All 24 candidate/cell summaries have
`failed=0`; actual rendered chat inputs equal the requested token counts and
share chat-template SHA-256
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.
An initial transcription used C=4 for the middle W1/W2/W3 cells. Those results
remain labeled as extra observations; C=8 was rerun and is the admitted value.

EP4 and EP8 each win six of twelve cells by measured e2e p99. Across the
single pass, median e2e p99 is 1356.06 ms for EP4 and 1334.32 ms for EP8;
median output throughput is 928.58 versus 881.51 output tokens/s. EP4 uses
four GPUs (peak selected-GPU memory sum 117546 MiB), while EP8 uses eight
(235898 MiB). Neither resource-aware candidate dominates the other, so both
advance. The auditable compact result is
`docs/Q-TopoMoE_Phase8_NVFP4_single_pass_screen_20260809.json`; raw request
JSONL and service logs remain under
`/data/models/test/qtopomoe_phase8_screen/` on gpu-111.

The transparent Phase 8 cost model was then replayed against all 12 measured
oracles. It chose EP4 in every cell because both configurations currently use
the same TP1 CUTLASS compute proxy and the measured mapping-rate term favors
TP4-NUMA. Top-1 accuracy is 0.5, median regret 0.33%, p95 regret 52.60%, and
controller overhead p95 0.028%. Therefore the p95-regret Gate fails and
`formal_gate_ready` remains false. This is kept as an explicit model-error
result in `docs/Q-TopoMoE_Phase8_NVFP4_selector_screen_replay_20260809.json`;
the next valid step is repeated service measurement and service-level
calibration, not hard-coding the twelve oracle answers or adding RL.

## Cross-format single-pass funnel (2026-08-09)

The same exact-token 12-cell screen now covers BF16 TP4, FP8 TP2, W4A16
TP1 x DP4, W4A16 static EP4, RedHat NVFP4 static EP4 and NVFP4 static EP8.
All six passed the request/token/topology Gates. The resource-aware Pareto set
is FP8 TP2, W4A16 EP4, NVFP4 EP4 and NVFP4 EP8. Detailed measurements,
failure recovery notes and the ordered next step are in
`docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.md`; the
machine-readable aggregate is the adjacent `.json` file. These are screening
results only and do not replace randomized five-repeat bootstrap-CI runs.

## Cross-format randomized repeated result (2026-08-10)

The four Pareto candidates completed all 20 seed-42 randomized service runs
and all 240 candidate/repetition/workload summaries. Request completion,
failed-zero, exact-token, common-chat-template and EP-rank-truth Gates all
passed. The accepted 10,000-resample aggregate and the uncertainty audit are
documented in
`docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.md`; the
adjacent `.json` retains per-cell medians, bootstrap 95% intervals and hashed
source-artifact references. FP8 TP2 wins 7/12 median-e2e cells, NVFP4 EP8 wins
4/12 and W4A16 EP4 wins 1/12. NVFP4 EP8 has materially higher repeated-run
variability, so service-level selector calibration must consume repeated
medians and uncertainty rather than a fastest-run oracle.

This completes the randomized repeated Gate for the current closed-loop
12-cell funnel. Although vLLM prefix caching was enabled, controlled prefix
hit ratios and Poisson/burst arrivals were not exercised. They remain a
separate formal matrix after transparent selector calibration and replay.

## Calibration readiness audit (2026-08-10)

The five-repeat medians and bootstrap intervals are now bound to the Phase 3
route trace and exact M-bucket workload records. Service observation and
uncertainty Gates pass for all four Pareto candidates. Calibration remains
blocked because the measured kernel DB lacks FP8 M=4/128 and all required
W4A16 WNA16 rows. The exact coverage audit and ordered remediation are in
`docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.md`. No synthetic or
cross-format latency is admitted to make the selector appear ready.

## CPU verification

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
python3 -m py_compile phase4/workload/generate_m_buckets.py selector/*.py scripts/analyze_phase1.py
```

No GPU process is started by these commands.

## Phase 7 topology/quantization-aware EPLB

`selector/eplb_policy.py` is the auditable policy layer.  The offline planner
requires explicit per-expert service load, exact quantized storage bytes, HBM
headroom, a measured pair-cost matrix, and the frozen trace SHA-256.  It emits
the expert-to-GPU mapping, replicas, predicted cross-NUMA bytes, dispatch-cost
proxy, migration bytes, per-GPU HBM use, predicted p99 proxy, and a stable plan
hash.  It does not mutate a serving runtime.

The online controller implements the frozen Runbook hysteresis defaults:
500 ms/1000-request windows, EMA alpha 0.2, load CV >0.25 for three windows,
minimum 5% benefit and 2x benefit/cost, ten-window residency, twenty-window
cooldown, and rollback after three windows above 5% p99 regression.

Use `scripts/audit_quantized_expert_sizes.py` to derive routed-expert bytes
from safetensors metadata and `scripts/build_eplb_plan.py` to build a plan.
The offline service-time and HBM inputs are now measured.  `formal_ready`
remains false until the live-service migration block/recovery/p99 Gate and
runtime placement-plan application both pass.
