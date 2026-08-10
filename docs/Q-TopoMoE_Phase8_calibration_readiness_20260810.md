# Phase 8 service-level calibration readiness (2026-08-10)

## Outcome

Calibration is correctly blocked on missing real micro-kernel measurements;
it is not blocked on service evidence. The accepted five-repeat aggregate was
converted into 12 Phase 8 observations. Every one contains all four candidate
median e2e-p99 measurements and the corresponding n=5, 10,000-resample
bootstrap 95% intervals.

The machine-readable audit is
`docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.json`. Its status is
`blocked_missing_calibration_inputs` and its Gates are:

| Gate | Result |
|---|---|
| all candidates have repeated service observations | pass |
| all candidates have bootstrap intervals | pass |
| all candidates have measured kernel coverage | **fail** |

No synthetic latency, cross-format substitution or oracle lookup was inserted.

## Kernel coverage

The exact Runbook workload records require M buckets
`1,4,8,16,32,128,256,2048,8192,16384`.

| Candidate | Cost backend | Missing M buckets | Ready |
|---|---|---|---|
| FP8 TP2 Triton | measured FP8 Triton | none | yes |
| W4A16 EP4 Triton | W4A16 runtime kernel | all ten | no |
| NVFP4 EP4 | measured NVFP4 CUTLASS proxy | none | yes |
| NVFP4 EP8 | measured NVFP4 CUTLASS proxy | none | yes |

The NVFP4 runtime reports Marlin/auto selection while its transparent compute
cost feature is the already measured native NVFP4 CUTLASS micro-kernel. This
proxy is explicit in the candidate manifest and still requires service-level
calibration; it is not represented as the runtime kernel itself.

## Evidence binding

- repeated observation file:
  `configs/strategies/phase8_observations_all_formats_repeated.json`
- observation SHA-256:
  `5f176527274306bc9d52692fe4599279e9851e026bf669f3a5b30cd9b9957157`
- Pareto candidate SHA-256:
  `ead089c58f03c8cbc4e86dd23ef584187d653be0569de628c40b4a7db0e29c3f`
- kernel DB SHA-256 after adding FP8 M=4/128:
  `0757a692a64e634bad280ea2eee7a72053217435f574ba4005a55f39150ced1d`
- Phase 3 route-token count: `119827`
- derived communication rate: `2827.5621954985104` bytes per trace token

The observation builder accepts both legacy single-pass scalars and repeated
bootstrap summaries. For repeated input it preserves medians and intervals in
the observation instead of collapsing to one arbitrary repetition.

## Ordered next action

FP8 M=4/128 is complete with the frozen cleanroom runtime. The valid 50-repeat
p50/p95 results are 216.31/233.25 us and 631.97/639.16 us respectively. Raw
compact evidence is
`docs/Q-TopoMoE_Phase4_triton_moe_fp8_missing_m_20260810.json` (SHA-256
`3f5a067d28358762b4c72f9e4043ace1ae219282dab115f0349582ce1a4aecc1`).
The first shell wrapper attempt stopped before Python because of local
PowerShell expansion of `$PATH`; it allocated no GPU memory and produced no
measurement. The corrected explicit `env PATH=...` launch produced the
admitted rows.

1. Add a W4A16 benchmark path that invokes the real vLLM WNA16 MoE kernel and
   its actual packed weight/scaling contract; do not benchmark BF16 tensors and
   label them W4A16.
2. Measure the ten required W4A16 M buckets, merge only valid finite-output
   rows into the kernel DB, and rerun this readiness Gate.
3. Only after all four candidates pass coverage, fit the transparent service
   correction and report held-out top-1, median/p95 regret and controller
   overhead. Do not add RL or hard-code the 12 oracle choices.
