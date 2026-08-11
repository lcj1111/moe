# Phase 8 — benchmark and calibration history

> Consolidated from the dated reports listed below. Source content is retained; only trailing whitespace was normalized. SHA-256 values are computed from the UTF-8 Git blob (LF-normalized); machine-readable artifacts keep their original paths for reproducibility.

## Source integrity

| Original file | UTF-8 bytes | SHA-256 of Git blob |
|---|---:|---|
| `docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.md` | 4882 | `BEFC17B978267D5F0AC7C9BBED648FD6B380C17FBA9E31AFB2FE16C0EE00D686` |
| `docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.md` | 4183 | `64EF6FE5C55120C8AEEB1ED9FD990C2BD1AA3CF360C4112DF8C8438810CA54A9` |
| `docs/Q-TopoMoE_Phase8_cache_arrival_pilot_v3_20260810.md` | 2526 | `1F13A9F2A2695BD4922988BC451B657E50F9250F2899F3633ACDE3EC88399780` |
| `docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.md` | 4570 | `2B54CDC8591ED12F13459A703F30CB66BB6F5A47323D3349E73F824B4E15A65E` |
| `docs/Q-TopoMoE_Phase8_capacity_rate_freeze_20260810.md` | 2462 | `30DA2FB452E868B293B1F21D75674677037C1F2C1FD0E32CB2EB0D27D799A790` |
| `docs/Q-TopoMoE_Phase8_repeated_run_incidents_20260809.md` | 1604 | `8D168CD632A928EDA2B2848EFE25BBC43AB24A984380D5ADD4DE59532E82135B` |
| `docs/Q-TopoMoE_Phase8_service_calibration_cv_20260810.md` | 3187 | `3D198C82056B37B0479B267386429B2EEDEAB280EBC91723BCD6312A3488B504` |

---

## Source: `docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.md`

# Phase 8 randomized five-repeat bootstrap result (2026-08-10)

## Result and admission Gate

The four candidates admitted by the cross-format single-pass funnel completed
20 independently started services in seed-42 randomized order: five service
repetitions per candidate and the exact 12-cell Runbook matrix per service.
All 240 candidate/repetition/workload summaries passed. The aggregate Gate is
`accepted`: all requests completed, `failed=0`, rendered input-token counts are
exact, the chat-template hash is identical, and actual EP rank counts equal the
expected 0/4/4/8 values.

The deterministic 10,000-resample aggregate is
`docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.json`. Raw
requests, GPU samples, metadata and server logs remain on gpu-111 under
`/data/models/test/qtopomoe_phase8_repeated_v4/`; they are not copied to Git.

Evidence identity:

- workload matrix SHA-256: `de91dd4a05fa610d73d2e8554ea4fabb7d4959f0423defc057c1190089a33e74`
- randomized schedule SHA-256: `9746e1873c9b504d99b10cf4e300cd250a8df1132cd331df923e6aed6c1e8740`
- chat-template SHA-256: `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`
- frozen vLLM: `0.26.1rc1.dev343+g33c50587d`
- frozen vLLM executable SHA-256: `ac1228d685d50a61a490694de01891653dffdbe4326fadfc6ed6cf39d9fde763`

## Candidate aggregate

Each latency/throughput column is the median across the 12 per-cell medians;
it is a compact comparison, not a workload-weighted production forecast.

| Candidate | GPUs | EP ranks | Median e2e p99 (ms) | Median output tok/s | Oracle wins | Peak selected-GPU memory sum (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| FP8 TP2 PIX 0,1 Triton | 2 | 0 | 1135.29 | 828.65 | 7 | 58878 |
| W4A16 static EP4 Triton | 4 | 4 | 1065.03 | 874.54 | 1 | 119652 |
| RedHat NVFP4 static EP4 NUMA0 | 4 | 4 | 1481.23 | 918.60 | 0 | 117540 |
| RedHat NVFP4 static EP8 SYS | 8 | 8 | 1249.63 | 1031.66 | 4 | 235896 |

All four remain in the resource-aware Pareto set because the comparison also
includes GPU count and peak selected-GPU memory. This does **not** mean all
four are equally suitable: FP8 TP2 is the default resource-efficient choice,
while NVFP4 EP8 is useful for the long-prefill/high-concurrency cells it wins.

## Per-cell median oracle

| Cell | Shape (input/output, concurrency) | Median-e2e-p99 oracle | Oracle median e2e p99 (ms) |
|---|---|---|---:|
| W1 C1 | 256/128, 1 | FP8 TP2 | 203.87 |
| W1 C8 | 256/128, 8 | FP8 TP2 | 861.76 |
| W1 C32 | 256/128, 32 | FP8 TP2 | 2427.48 |
| W1 C128 | 256/128, 128 | FP8 TP2 | 5093.57 |
| W2 C1 | 2048/256, 1 | FP8 TP2 | 306.79 |
| W2 C8 | 2048/256, 8 | W4A16 EP4 | 904.68 |
| W2 C32 | 2048/256, 32 | NVFP4 EP8 | 1971.10 |
| W3 C1 | 8192/256, 1 | NVFP4 EP8 | 576.79 |
| W3 C8 | 8192/256, 8 | FP8 TP2 | 879.34 |
| W3 C16 | 8192/256, 16 | NVFP4 EP8 | 1482.24 |
| W4 C1 | 32768/128, 1 | NVFP4 EP8 | 1451.90 |
| W4 C4 | 32768/128, 4 | FP8 TP2 | 524.85 |

The oracle uses the median of the five measured e2e-p99 values. Every JSON
row also contains a non-parametric bootstrap 95% interval for e2e p99, TTFT
p99, TPOT p99 and output throughput, plus paths and SHA-256 values for all
five source summaries.

## Repeat variability and interpretation

Median cell-level e2e-p99 coefficient of variation is 0.86% for FP8 TP2,
1.04% for W4A16 EP4, 5.55% for NVFP4 EP4 and 18.02% for NVFP4 EP8. The worst
NVFP4 EP8 cell is W1 C1: 269.84--551.20 ms across the five runs (2.04x).
W4A16 EP4 also shows a 1.42x range at W1 C32. These observations remain valid
measurements, but selector calibration must account for their uncertainty and
must not learn directly from a single fastest repetition.

With only five independent service repetitions, the intervals are an
uncertainty indicator rather than proof of a stable population tail. A wider
follow-up should be triggered for decisions whose candidate intervals overlap
materially, especially for NVFP4 EP8.

## Scope boundary and ordered next step

The frozen services report `enable_prefix_caching=True`, but this workload did
not construct or verify controlled 0%/50%/90% prefix-hit populations. The
client is closed-loop; Poisson and burst arrivals were not generated. Thus
this result completes the Runbook's randomized five-repeat/bootstrap Gate for
the current 12-cell funnel, but it does not complete the formal prefix-cache x
arrival-mode matrix.

The next ordered engineering step is service-level Phase 8 selector
calibration against these repeated medians and uncertainty bounds, followed by
replay of top-1/median-regret/p95-regret/controller-overhead Gates. It must use
transparent coefficients and held-out validation; it must not hard-code the
12 oracle answers or introduce RL. After that calibration is auditable, add
controlled prefix reuse and closed-loop/Poisson/burst traffic generation and
run the remaining formal matrix on the retained candidates.

---

## Source: `docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.md`

# Phase 8 cross-format single-pass screen (2026-08-09)

## Gate and scope

The frozen 12-cell Runbook matrix was executed once for six representative
BF16, FP8, W4A16 and NVFP4 configurations. Every candidate completed all 12
cells with `failed=0`, `completed=requests`, exact rendered input-token counts,
and chat-template SHA-256
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.
The W4A16 EP4 and NVFP4 EP4/EP8 logs and metadata report actual EP ranks 4, 4
and 8 respectively. Raw requests, GPU samples and server logs remain in
`/data/models/test/qtopomoe_phase8_screen/` on gpu-111. The compact, hashed
result is `docs/Q-TopoMoE_Phase8_all_formats_single_pass_screen_20260809.json`.

This is a screening pass, not a confidence-interval result. Throughput is the
client's completed output-token count divided by the request-batch wall time.

## Aggregate results

| Candidate | GPUs | Median e2e p99 (ms) | Median output tok/s | Oracle wins | Peak selected-GPU memory sum (MiB) | Pareto |
|---|---:|---:|---:|---:|---:|---|
| BF16 TP4 NUMA0 | 4 | 1471.14 | 831.80 | 1 | 117814 | no |
| FP8 TP2 PIX 0,1 Triton | 2 | 1220.20 | 776.34 | 7 | 58826 | yes |
| W4A16 TP1 x DP4 Triton | 4 | 1236.89 | 750.84 | 0 | 119528 | no |
| W4A16 static EP4 Triton | 4 | 1157.84 | 850.65 | 1 | 119570 | yes |
| RedHat NVFP4 static EP4 | 4 | 1356.06 | 928.58 | 0 | 117546 | yes |
| RedHat NVFP4 static EP8 | 8 | 1334.32 | 881.51 | 3 | 235898 | yes |

The resource-aware Pareto set therefore advances FP8 TP2, W4A16 EP4, NVFP4
EP4 and NVFP4 EP8. BF16 TP4 and W4A16 DP4 remain preserved as measured
baselines but do not advance to the repeated Pareto run.

FP8 TP2 won W1 at C=1/8/32/128, W2 C=1, W3 C=8 and W4 C=4. NVFP4 EP8 won
W2 C=8/32 and W4 C=1. BF16 TP4 won W3 C=1, and W4A16 EP4 won W3 C=16.

## Runtime findings

The first FP8 `auto`-backend launch was rejected before serving: DeepGEMM
raised `Unknown SF transformation` while transforming FP8 block scales. The
failure is retained under
`/data/models/test/qtopomoe_phase8_screen/fp8_tp2_pix01_phase8_screen_v1` and
is not included in the aggregate. Relaunching the frozen checkpoint with
`VLLM_MOE_BACKEND=triton` selected the native vLLM Triton FP8 MoE backend and
passed all 12 cells. No framework or checkpoint hotpatch was used.

The W4A16 DP4 service emitted `EngineDeadError` only after the runner's
deliberate shutdown, after all workload summaries had been written and every
inference request returned HTTP 200. It is classified as a shutdown artifact,
not an inference failure.

## Repeated-run follow-up

This ordered step is complete. The four Pareto candidates ran in seed-42
randomized order with five independent service repetitions, health check,
warmup, measurement, clean stop and cooldown. The accepted 10,000-resample
result and interpretation are in
`docs/Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.md`; its
machine-readable evidence is the adjacent `.json` file. Prefix-cache
0/50/90% and closed-loop/Poisson/burst still require controlled formal runs.

The frozen repeated-run implementation is
`configs/experiments/phase8_pareto_repeated_v1.json` plus
`scripts/run_phase8_repeated.py`. It creates a seed-42 schedule of 20 service
runs, refuses to overwrite incomplete evidence, verifies idle selected GPUs,
pins the P2P-enabled environment, rejects any vLLM version other than the
cleanroom-frozen `0.26.1rc1.dev343+g33c50587d`, checks the exact runtime
backend and EP rank count after warmup, audits every cell, and cools down for
30 seconds. Run it on gpu-111 with:

```bash
/home/k8s-ops/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129/bin/python \
  scripts/run_phase8_repeated.py \
  --plan configs/experiments/phase8_pareto_repeated_v1.json \
  --output-root /data/models/test/qtopomoe_phase8_repeated_v1
```

All 20 runs passed and deterministic intervals were generated with
`scripts/aggregate_phase8_repeated.py`. Raw artifacts remain on the data
volume; only the compact aggregate is committed.

Pre-measurement runner corrections and their preserved evidence directories
are recorded in `docs/Q-TopoMoE_Phase8_repeated_run_incidents_20260809.md`.

---

## Source: `docs/Q-TopoMoE_Phase8_cache_arrival_pilot_v3_20260810.md`

# Phase 8 controlled cache/arrival pilot v3 — accepted

This pilot validates workload mechanics only. It must not be used as a deployment ranking or as a service-model calibration dataset: it uses one FP8 TP2 candidate, 12 requests per cell, 4224 input tokens and 16 output tokens.

## Outcome

- 9/9 cells completed; 108/108 requests completed; failed=0.
- Every server-reported prompt length was exactly 4224 tokens.
- `prompt_tokens_details.cached_tokens` was present for every measured request.
- All cache-ratio and arrival-schedule Gates passed.
- Only GPU0-1 were selected; the runner released them after completion.
- Raw evidence: `/data/models/test/qtopomoe_phase8_cache_arrival_pilot_v3_20260810`.

| Semantic shared prefix | Engine-realizable cache ratio | Measured ratio | Result |
|---:|---:|---:|---|
| 0% | 0% | 0% | pass |
| 50% | 50% | 50% | pass |
| 100% | 75% | 75% | pass |

The 100% semantic prefix is not a 100% cache hit. The frozen vLLM source caps the hit length at `prompt_length - 1`, because at least the final prompt token must be computed to obtain logits. Qwen3.5 MoE also aligns its hybrid attention/Mamba cache page to 1056 tokens. For the exact 4224-token pilot, the maximum hit is therefore `floor((4224 - 1) / 1056) * 1056 = 3168`, or 75%.

The two rejected pilots were useful hard-Gate discoveries, not accepted results:

1. v1 used 256-token prompts. They are shorter than one 1056-token page, so 50% and 100% semantic sharing both yield zero cache hits.
2. v2 validated the 50% hit but exposed the `prompt_length - 1` rule. Its fixed 4 rps also overloaded the zero-cache population.
3. v3 used block-aware expectations and a common 2.0 rps open-loop rate derived as 70% of the v2 zero-cache closed-loop throughput.

For the six Poisson/burst cells, p95 request-start lag versus the frozen schedule ranged from 0.289 ms to 1.644 ms, against a 125 ms Gate. The client records nominal rate, realized finite-sample schedule rate, actual start rate, scheduled/realized inter-arrivals, and peak in-flight requests separately.

## Next admissible step

Build the formal matrix in two stages: first measure the closed-loop capacity of each base cell, then freeze a candidate-independent open-loop rate from the slowest admissible candidate/cell times a safety utilization. Expand each base cell across semantic prefix populations and arrival modes while retaining the 1056-token page and `L-1` rules. Only the repeated formal results should be used to refit the transparent Phase 8 selector.

---

## Source: `docs/Q-TopoMoE_Phase8_calibration_readiness_20260810.md`

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

---

## Source: `docs/Q-TopoMoE_Phase8_capacity_rate_freeze_20260810.md`

# Phase 8 controlled-capacity audit and formal rate freeze (2026-08-10)

Status: accepted. This is a load-control Gate, not a candidate ranking or
selector-calibration result.

The controlled p0 closed-loop capacity prepass completed all four candidates,
three repetitions per candidate, and twelve Runbook base cells. All 12 runs
have `status=workload_passed`; all 144 summaries have `failed=0`, complete
requests, exact client and server prompt tokens, complete cache usage details,
zero actual cached-token ratio, and a passing closed-loop arrival Gate.

The formal open-loop rate for each base cell is the minimum observed request/s
across all four candidates and all three repetitions, multiplied by 0.70 and
floored to six decimal places. Poisson and burst use the same frozen rate for
the same base cell; closed-loop remains rate-free. This prevents a strategy
from being ranked under an easier candidate-specific offered load.

| Base cell | Minimum capacity (req/s) | Frozen rate (req/s) | Limiting candidate |
|---|---:|---:|---|
| w1_c1 | 2.547209 | 1.783046 | NVFP4 EP8 |
| w1_c8 | 7.157322 | 5.010125 | W4A16 EP4 |
| w1_c32 | 13.217492 | 9.252244 | W4A16 EP4 |
| w1_c128 | 17.439556 | 12.207689 | W4A16 EP4 |
| w2_c1 | 2.087605 | 1.461323 | W4A16 EP4 |
| w2_c8 | 4.455057 | 3.118539 | W4A16 EP4 |
| w2_c32 | 5.134312 | 3.594018 | FP8 TP2 |
| w3_c1 | 1.261208 | 0.882845 | FP8 TP2 |
| w3_c8 | 1.538196 | 1.076737 | FP8 TP2 |
| w3_c16 | 1.616130 | 1.131291 | FP8 TP2 |
| w4_c1 | 0.370480 | 0.259336 | FP8 TP2 |
| w4_c4 | 0.373900 | 0.261729 | FP8 TP2 |

The generated formal matrix contains 108 cells: 12 base cells x semantic
prefix populations 0/50/100 x closed-loop/Poisson/burst. All 108 stream seeds
are unique. Cache page size remains runtime-discovered per candidate rather
than hard-coded: FP8 TP2 reported 1056 tokens, W4A16 EP4 and NVFP4 EP4 528,
and NVFP4 EP8 272 under the frozen 65K service configuration.

Raw source root:
`/data/models/test/qtopomoe_phase8_capacity_prepass_v1_20260810`

Evidence SHA-256:

- capacity audit: `c5a1e98acdf74b710a388d460acc8e00817832f8a970e0815a3461257fd35005`
- formal workload matrix: `3fc04a14f481cd008ee6ed6b556d020edf2c3cee4bf14e732d0808878710852e`

The next admissible action is a frozen-runtime dry-run of
`configs/experiments/phase8_formal_controlled_v1.json`, followed by the
randomized five-repeat formal service matrix only if the schedule, GPU-idle,
runtime-version and matrix-SHA Gates pass.

---

## Source: `docs/Q-TopoMoE_Phase8_repeated_run_incidents_20260809.md`

# Phase 8 repeated-run pre-measurement incidents (2026-08-09)

No result from the three directories below is admitted to the repeated
comparison. Each attempt stopped before a workload cell was measured, kept its
failure metadata and logs, and released all selected GPUs.

| Attempt | Evidence on gpu-111 | Gate result | Root cause | Resolution |
|---|---|---|---|---|
| v1 | `/data/models/test/qtopomoe_phase8_repeated_v1` | rejected before health | runner default selected legacy vLLM 0.26.0 instead of the frozen 0.26.1rc build | exact cleanroom vLLM version and entrypoint SHA Gate, commit `6e790b3` |
| v2 | `/data/models/test/qtopomoe_phase8_repeated_v2` | rejected during engine startup | absolute venv entrypoint did not prepend its `bin` directory to `PATH`; FlashInfer JIT could not find the already-installed `ninja` | frozen venv PATH activation plus ninja path/SHA Gate, commit `c941881` |
| v3 | `/data/models/test/qtopomoe_phase8_repeated_v3` | FP8 rejected after a successful 32-request warmup | backend Gate expected quotes that the official log does not emit; actual line was `Using TRITON Fp8 MoE backend` | stable exact substring plus forbidden `Unknown SF transformation` Gate, commit `e4ee0b8` |

Before the v3 FP8 text-Gate rejection, NVFP4 EP8, W4A16 EP4 and NVFP4 EP4
each completed 12/12 cells with zero request failures. They are intentionally
not merged into the final dataset because v4 is a clean, single-plan restart.

The admitted run root is `/data/models/test/qtopomoe_phase8_repeated_v4`.
Its schedule records frozen vLLM and ninja hashes before launching any service.

---

## Source: `docs/Q-TopoMoE_Phase8_service_calibration_cv_20260810.md`

# Phase 8 transparent service calibration CV (2026-08-10)

## Outcome

The transparent calibration was executed and correctly failed the Runbook
regret Gates. It is not accepted for deployment.

| Evaluation | Top-1 | Median regret | p95 regret | Overhead p95 |
|---|---:|---:|---:|---:|
| uncalibrated 12-cell replay | 33.33% | 33.89% | 101.80% | 0.04% |
| grouped held-out calibration | 41.67% | 13.64% | 54.77% | 0.05% |
| required Gate | report | <=5% | <=10% | <1% |

Only the controller-overhead Gate passes. The machine-readable result is
`docs/Q-TopoMoE_Phase8_service_calibration_cv_20260810.json`; the frozen
uncalibrated baseline is
`docs/Q-TopoMoE_Phase8_uncalibrated_replay_20260810.json`.

## Method and leakage control

The base cost remains measured MoE kernel time, measured mapping communication
cost, route imbalance and migration cost. Kernel invocation accounting uses
40 model layers and, per request, one prefill invocation plus `output_tokens`
decode invocations. Calibration adds only a candidate-specific non-negative
scale and intercept.

Fits use weighted least squares. Bootstrap 95% intervals are converted to an
approximate variance with a 5% median sigma floor so an artificially narrow
five-run interval cannot dominate. W1, W2, W3 and W4 are each held out in turn;
the held-out family's service medians and oracle labels never enter that fold's
fit. Every one of the 12 observations is evaluated exactly once out of family.

The initial normalized-M-histogram implementation was also audited. It counted
only one average kernel invocation and gave 19.31% median / 70.52% p95 regret.
Correct sequence invocation counting improved those figures to 13.64% / 54.77%
but did not make the result acceptable.

## Diagnosis

The residual is structured, not random fit noise. The current repeated screen
has `enable_prefix_caching=True`, but it did not construct or verify controlled
0%, 50% and 100% prefix-hit populations. All requests are closed-loop; Poisson
and burst arrivals are absent. Consequently queueing, batching and implicit
prefix reuse are not identifiable from the current feature set.

The strongest symptom is W4 (32768 input / 128 output): measured e2e p99 for
FP8 falls from 2131.19 ms at C=1 to 524.90 ms at C=4, while the nominal prefix
field is unchanged. Similar candidate-rank reversals occur across families.
A more flexible regression could memorize these 12 cells, but that would not
be a valid selector and is explicitly rejected.

Full-data coefficients are retained in the JSON only for audit and diagnosis.
They are not an accepted deployment manifest.

## Ordered next action

1. Construct requests with measured 0%, 50% and 100% prefix reuse and record
   rendered-token and prefix-identity hashes.
2. Add deterministic closed-loop, Poisson and burst arrival generators and
   record realized inter-arrival statistics.
3. Run the formal crossed matrix on retained candidates, preserving repeated
   service starts and bootstrap uncertainty.
4. Refit the same transparent model with explicit prefix-hit and arrival-mode
   features, then repeat grouped held-out regret Gates. Do not add RL or encode
   per-cell oracle choices.
