# Qwen3.5 MoE canonical W4A16 TP2 smoke (2026-08-05)

Both native backends passed 32/32 requests with no failures on PIX GPUs 0–1.
The fixed smoke used 256 requested input tokens, 64 output tokens, concurrency
8, and seed 42.

| Backend | TTFT p50/p95 | TPOT p50/p95 | E2E p50/p95 | Wall time |
|---|---:|---:|---:|---:|
| vLLM | 265/779 ms | 71.4/81.1 ms | 4.76/5.89 s | 5.89 s |
| SGLang | 251/7341 ms | 63.7/66.4 ms | 4.27/11.33 s | 11.33 s |

This is a functionality smoke, not a formal ranking. SGLang's TTFT long tail
contains first-run compilation/scheduling effects and requires warmup plus
repeated runs.

## Memory interpretation

Matched TP2 logs do not show a large weight-memory difference:

- vLLM per GPU: 10.15 GiB weight/non-Torch plus 17.91 GiB hybrid cache.
- SGLang per GPU: 10.08 GiB weight, 8.25 GiB Mamba state cache, and
  9.18 GiB KV cache.

SGLang also reserved up to 1 GiB of multimodal CUDA IPC memory on the base GPU
because automatic multimodal transport remained enabled. Future text-only Gates
pass `--language-only` to remove this irrelevant reservation. Fair formal memory
comparison must fix cache capacity or maximum concurrency, not only set both
framework-specific memory-fraction arguments to the same number.
