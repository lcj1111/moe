# Phase 8 service-level calibration readiness (2026-08-10)

## Outcome

Calibration inputs are ready. The accepted five-repeat aggregate was converted
into 12 Phase 8 observations. Every one contains all four candidate median
e2e-p99 measurements and the corresponding n=5, 10,000-resample bootstrap 95%
intervals. The final missing W4A16 rows were measured through the real packed
int4 vLLM Triton WNA16 MoE path and admitted only after a dequantized numerical
reference Gate and finite-output checks.

The machine-readable audit is
`docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.json`. Its status is
`ready` and its Gates are:

| Gate | Result |
|---|---|
| all candidates have repeated service observations | pass |
| all candidates have bootstrap intervals | pass |
| all candidates have measured kernel coverage | pass |

No synthetic latency, cross-format substitution or oracle lookup was inserted.

## Kernel coverage

The exact Runbook workload records require M buckets
`1,4,8,16,32,128,256,2048,8192,16384`.

| Candidate | Cost backend | Missing M buckets | Ready |
|---|---|---|---|
| FP8 TP2 Triton | measured FP8 Triton | none | yes |
| W4A16 EP4 Triton | measured WNA16 Triton | none | yes |
| NVFP4 EP4 | measured NVFP4 CUTLASS proxy | none | yes |
| NVFP4 EP8 | measured NVFP4 CUTLASS proxy | none | yes |

The W4A16 service still uses runtime `kernel_backend=triton`; its cost rows use
the more precise `cost_kernel_backend=wna16_triton` namespace so they cannot be
confused with BF16/FP8 Triton rows. The NVFP4 runtime reports Marlin/auto
selection while its transparent compute
cost feature is the already measured native NVFP4 CUTLASS micro-kernel. This
proxy is explicit in the candidate manifest and still requires service-level
calibration; it is not represented as the runtime kernel itself.

## Evidence binding

- repeated observation file:
  `configs/strategies/phase8_observations_all_formats_repeated.json`
- observation SHA-256:
  `5f176527274306bc9d52692fe4599279e9851e026bf669f3a5b30cd9b9957157`
- Pareto candidate SHA-256:
  `cf2fb1e23025f8aa2456a5086bd365c768a20b8ca66d04c8b0bc6f06a772161a`
- kernel DB SHA-256 after adding all W4A16 buckets:
  `6ae396bff2ec91156a8c2b9bbf60b7ee14af5d198f42b3c593680e69b82302e8`
- W4A16 compact evidence SHA-256:
  `8866c5f79f1c4720d0bdc1a2f27af739b86efb321ba5df9085a53a53b7c056b0`
- Phase 3 route-token count: `119827`
- derived communication rate: `2827.5621954985104` bytes per trace token

The observation builder accepts both legacy single-pass scalars and repeated
bootstrap summaries. For repeated input it preserves medians and intervals in
the observation instead of collapsing to one arbitrary repetition.

## W4A16 measurement admission

FP8 M=4/128 is complete with the frozen cleanroom runtime. The valid 50-repeat
p50/p95 results are 216.31/233.25 us and 631.97/639.16 us respectively. Raw
compact evidence is
`docs/Q-TopoMoE_Phase4_triton_moe_fp8_missing_m_20260810.json` (SHA-256
`3f5a067d28358762b4c72f9e4043ace1ae219282dab115f0349582ce1a4aecc1`).
The first FP8 shell wrapper attempt stopped before Python because of local
PowerShell expansion of `$PATH`; it allocated no GPU memory and produced no
measurement. The corrected explicit `env PATH=...` launch produced the
admitted rows.

The W4A16 measurement uses `vllm.fused_experts` with `use_int4_w4a16`, BF16
activations, group size 128, uint8 N-first weights containing two centered
int4 nibbles per byte, and BF16 group scales. A deterministic small-shape Gate
compares the packed kernel to explicit dequantization and passed with maximum
absolute error `7.6195e-06`. All ten real-geometry buckets passed 50 repeats,
shape and finite-output Gates. p50 ranges from 151.48 us (M=1) to 17546.67 us
(M=16384). Compact evidence is
`docs/Q-TopoMoE_Phase4_triton_moe_w4a16_required_m_20260810.json`.

The frozen vLLM build has no device-specific tuned config file for this WNA16
shape, so it reports use of the official default MoE config and warns that
performance may be sub-optimal. The measurements remain real and admissible,
but future kernel tuning must be treated as a separate optimization stage.

## Ordered next action

1. Fit the transparent service-level correction from repeated medians while
   preserving a held-out workload split and bootstrap uncertainty.
2. Replay all held-out observations and report top-1, median/p95 regret and
   controller overhead against the Runbook Gates.
3. If a Gate fails, diagnose the missing feature or split instability; do not
   hard-code the 12 oracle choices or introduce RL.
