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
