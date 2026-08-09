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

## Next ordered step

Run the four Pareto candidates in seeded randomized order, five independent
service repetitions per candidate and workload cell, with health check,
warmup, measurement, clean stop and 30-second cooldown for every repetition.
Report bootstrap 95% confidence intervals. Prefix-cache 0/50/90% and the
closed-loop/Poisson/burst traffic dimensions still require their formal runs;
the current screen only establishes the candidate funnel.
