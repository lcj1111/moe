# Qwen3.5 MoE W4A16 compatibility baseline (2026-08-05)

## Frozen checkpoint

- Path: `/data/models/test/qtopomoe_w4a16`
- Architecture: `Qwen3_5MoeForCausalLM`
- Model type: `qwen3_5_moe_text`
- Quantization: `compressed-tensors` W4A16
- Tensor count: 93,093
- `model.safetensors`: 20,928,221,896 bytes
- `model.safetensors` SHA-256:
  `36e5b5ec77c5c35e3ce23f415e31c7fcbd5b14e287f02629ce057698cdd6d94f`
- Full eight-file manifest: `docs/Q-TopoMoE_W4A16_freeze_20260805.json`

The checkpoint has a mixed export identity: its config is text-only, while
93,092 of 93,093 tensor names remain under `model.language_model.*`. The only
root tensor is `lm_head.weight`. This is an export/layout compatibility issue;
it is not evidence that GPTQ/W4A16 arithmetic is invalid.

## Frozen serving environments

| Backend | Framework | Transformers | compressed-tensors | Torch | FlashInfer |
|---|---:|---:|---:|---:|---:|
| vLLM | 0.26.0 | 5.14.1 | 0.17.0 | 2.11.0 | 0.6.14 |
| SGLang | 0.5.16 | 5.12.1 | 0.17.2a20260728 | 2.11.0 | 0.6.14 |

Both framework versions are the latest stable versions available from the
configured PyPI index on 2026-08-05. Upstream clean-room environments are now
also frozen for the next Gate:

- vLLM exact official nightly wheel/source commit:
  `33c50587d2679ba9bacc2a51ae19901f7eb3a129`
  (`0.26.1rc1.dev343+g33c50587d`, Torch `2.13.0+cu130`)
- SGLang source commit: `6c05aaae7e3966469b6c552aa11b545e5d27f8bf`
  (`0.5.17.dev0+g6c05aaae7e`, `sglang-kernel==0.4.5`,
  FlashInfer `0.6.15.post1`, Torch `2.11.0+cu130`)
- Transformers reference: `d24d79da55f7ee6e538a460d3025e41dcc41ab21`

The initially recorded vLLM source reference had no matching precompiled wheel.
It was therefore not overlaid on the released environment. The selected vLLM
commit is the exact commit advertised by the official wheel index, avoiding a
Python-source/CUDA-binary mismatch. SGLang is built from its pinned source with
the upstream-supported `SGLANG_BUILD_RUST_EXTS=none`; its CUDA serving kernels
remain the pinned packaged dependencies.

## Evidence-based status

### vLLM development adapter

The development-only adapter passed `/health`, `/v1/models`, a real chat
completion (`42`), and `/metrics`. It also logged native compressed-tensors,
Marlin linear, and Marlin MoE selection. This proves smoke-level numerical
usability, but it is not the final deployment because it used vLLM's
development/debug class override and a launcher-time internal config hook.

Evidence directory:
`/data/models/test/qtopomoe_w4a16_runs/vllm_adapter_tp1_hybrid/acceptance`

### SGLang development adapter

Rejected. The attempted adapter loaded only about 17.53 GiB and emitted
`Parameter ... not found` for `lm_head` and multiple GDN projection/output
weights, then failed CUDA Graph capture with an attention-backend interface
error. It must not be treated as a valid load or serving result.

Evidence directory:
`/data/models/test/qtopomoe_w4a16_runs/sglang_adapter_tp1_weights`

## Correct continuation policy

1. Do not add more compatibility fields to the rejected adapter.
2. Test pinned upstream framework sources in isolated environments first.
3. Require zero missing/skipped checkpoint parameters.
4. If the frozen checkpoint still fails, create a separate canonical serving
   checkpoint by deterministic tensor-key conversion; never overwrite the
   frozen source.
5. Accept a backend only after health, discovery, real completion, metrics,
   parameter coverage, and BF16 quality comparison all pass.

