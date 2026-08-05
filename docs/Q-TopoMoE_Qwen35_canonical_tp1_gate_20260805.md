# Qwen3.5 MoE canonical W4A16 TP1 Gate (2026-08-05)

The value-identical canonical checkpoint passed native vLLM and SGLang TP1
serving without an adapter, `PYTHONPATH` injection, or model-class override.

| Backend | Native quantized path | Weight load | GPU weight memory | Health/models/completion/metrics | Coverage warnings |
|---|---|---:|---:|---|---:|
| vLLM | Marlin linear + Marlin WNA16 MoE | 21.40 s | 19.53 GiB | pass (`42`) | 0 bytes |
| SGLang | CompressedTensors WNA16 Marlin MoE | 19.00 s | 19.75 GiB | pass (`42`) | 0 bytes |

Checkpoint:

- Path: `/data/models/test/qtopomoe_w4a16_canonical_text_v1`
- Weight SHA-256:
  `0eb2775989321d038ea9534041a6030f536a0d0d2293e7087837390e9738c1d2`
- Tensor equality: 93,093/93,093 tensors; shape, dtype, and complete value
  equality verified over 20,915,187,456 tensor bytes
- Config changed: no

Evidence:

- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_canonical/vllm_33c505_tp1_retry1`
- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_canonical/sglang_6c05_tp1`

This closes the architecture/weight-layout compatibility Gate. It does not yet
close BF16 quality comparison, repeated performance benchmarking, TP2/TP4/TP8,
or CUDA Graph production-mode validation.
