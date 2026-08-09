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
