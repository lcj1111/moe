# Phase 3 NVFP4 route trace and Phase 7 EPLB checkpoint

## Accepted route-trace Gate

The RedHatAI NVFP4 checkpoint was captured with the same frozen protocol as
BF16/W4A16: 116 prompts, manifest order 0..115, seed 42, 16 generated tokens,
TP4 on GPU0-3 and `max_model_len=8192`.  Runtime evidence confirms the native
`VLLM_CUTLASS` MoE backend.

- prompt count: 116/116
- prompt ID/order, prompt-token SHA-256 and prompt-token counts: all match BF16
- token positions: 119,827
- layer-token rows: 4,793,080 (`[token, 40, 8]`, `uint8`)
- expert range: 0..255
- 116 array SHA-256 values: all match the capture manifest
- `routed_experts is None`: absent
- capture manifest SHA-256: `114c2180d15d85b94a96b4c6914c24496f4422bb0a400e9e21d7d5646efdd4ab`

BF16 to NVFP4 route drift is `mean_jaccard=0.794115`,
`mean_flip_rate=0.660875`, and `mean_count_correlation=0.99898`.  Quantization
therefore changes many token-level top-k memberships while preserving the
aggregate expert-load shape very closely.  This is an input to placement, not
a replacement for the already accepted quality Gate.

The real workload mix derived from the trace is M=1: 0.007198, M=2048:
0.897493 and M=8192: 0.095309.

## Quantization-aware expert bytes

Safetensors metadata was audited without loading model tensors.  All 40x256
routed experts are present.  Each expert contains 12 stored tensors and is
exactly 1,769,496 bytes; total routed-expert storage is 18,119,639,040 bytes.
The audit includes packed U8 weights, FP8 scales and FP32 global scales rather
than estimating storage from nominal 4-bit weights.

## Topology and migration primitives

`configs/experiments/topology_gpu111.yaml` now records the live P2P state:
all off-diagonal read/write pairs are `OK`.  Formal NCCL five-run data remains
the mapping source; NODE-preferred groups are intentionally retained because
the host's measured NCCL result is better than the PIX negative control.

Exact-size (1,769,496-byte) migration microbench, five repeats:

| primitive | mean us | p95 us | effective GB/s |
|---|---:|---:|---:|
| logical same-GPU remap | 0.0421 | 0.0424 | n/a |
| same-GPU copy | 7.8008 | 7.8525 | 226.84 |
| same-NUMA PIX 0->1 | 42.6011 | 42.7145 | 41.54 |
| same-NUMA NODE 0->2 | 43.1823 | 43.4161 | 40.98 |
| cross-NUMA SYS 0->4 | 66.1044 | 66.6786 | 26.77 |

All copies verified content and reported peer access enabled.  These are
primitive costs only; block time, recovery time, and affected service p99 are
still required before online migration is accepted.

## Native NVFP4 kernel and EP admission

The native vLLM CUTLASS NVFP4 MoE primitive was measured for the real trace
buckets with 50 repeats: p95 is 250.42 us at M=1, 728.23 us at M=2048, and
2219.58 us at M=8192.  Applying the frozen token-weighted M mixture and
dividing by `M * top_k` yields an auditable offline placement proxy of
0.268434768470764 us per routed assignment.  It is not an end-to-end service
latency.

Official RedHatAI NVFP4 passed both real expert-parallel admission cells:

| cell | actual EP ranks | backend | completed/failed | e2e p50/p95 ms |
|---|---:|---|---:|---:|
| TP4 + EP4 static | 4 | MARLIN | 32/0 | 751.95 / 1682.29 |
| TP8 + EP8 static | 8 | MARLIN | 32/0 | 823.76 / 1996.98 |

The backend change from `VLLM_CUTLASS` in non-EP serving to `MARLIN` in the
sharded EP cells is runtime-selected and explicitly recorded; CUTLASS
microbench rows are not presented as EP MARLIN timings.  EP8 peak HBM is
29027--29109 MiB per GPU.  After subtracting the exact 1280 local routed
experts per GPU, measured non-expert/KV headroom is 26.24--26.32 GiB.

## EPLB implementation status

`selector/eplb_policy.py` now provides:

- deterministic offline placement using per-expert load, exact quantized
  bytes, HBM headroom, source-GPU weights, and measured pair costs;
- optional redundant-expert placement;
- predicted cross-NUMA bytes, dispatch cost, HBM use, migration bytes, and a
  stable plan SHA-256;
- the Runbook online state machine (500 ms/1000 requests, EMA 0.2, CV >0.25
  for three windows, benefit >=5%, benefit/cost >=2, residency 10, cooldown
  20, and rollback after three >5% p99-regression windows).

The measured-input full-domain plan covers 10,240 experts.  Its offline inputs
are ready: exact expert bytes, accepted route load, measured kernel proxy,
measured EP8 HBM headroom, and measured topology are all bound to the plan.
The load span is below 0.81 us across eight GPUs and the stable plan SHA-256 is
`d53bb6653abed0fe67163888dd838a8f2aef60d2d86968cc89f0c3d0430865d6`.
Overall `formal_ready` remains false until live-service migration
block/recovery/p99 impact and runtime placement-plan application pass.

The native vLLM EPLB admission cell was also executed with a real TP8/EP8
world.  It failed during model construction, before serving, with
`NotImplementedError: EPLB is not supported
CompressedTensorsW4A4Nvfp4MoEMethod.`  The frozen vLLM build and current
upstream main both leave EPLB disabled for this compressed-tensors NVFP4
method.  A historical upstream implementation supports the different
`ModelOptNvFp4FusedMoE` path; it is not evidence that changing the capability
property for the current path is safe.  The project therefore does not
hot-patch the serving venv.  Static EP4/EP8 remain admitted, while native EPLB
and custom runtime plan application are marked unavailable/pending rather
than reported as successful.
