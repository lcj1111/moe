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
