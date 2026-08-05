# Q-TopoMoE Phase 2 quality smoke and backend gate

Date: 2026-08-05 CST

## Frozen controls

- Quality set: `/data/models/test/qtopomoe_quality/frozen/quality_smoke_v1.jsonl`
- Frozen rows: 164 (GSM8K 32, MMLU-Pro 64, C-Eval 52, HumanEval 16)
- Automatically scored rows: 148. HumanEval is not executed outside an isolated sandbox.
- All runs use seed 42, concurrency 8, TP4 on GPUs 0-3, max model length 4096, and the exact vLLM cleanroom commit `33c50587d2679ba9bacc2a51ae19901f7eb3a129` unless stated otherwise.

## Results

| Runtime | MoE backend | GSM8K | C-Eval | MMLU-Pro | Requests |
|---|---|---:|---:|---:|---:|
| BF16 vLLM | Triton unquantized | 96.88% | 86.54% | 42.19% | 148/148 |
| W4A16 vLLM | Marlin | 59.38% | 36.54% | 20.31% | 148/148 |
| W4A16 vLLM | Triton WNA16 | 93.75% | 82.69% | 37.50% | 148/148 |
| W4A16 SGLang | native WNA16 Marlin | 93.75% | 78.85% | 40.63% | 148/148 |

The W4A16 vLLM Marlin result fails the 10-point smoke threshold on all three
benchmarks. The same checkpoint passes when only the vLLM MoE backend is
changed to `triton`: drops versus vLLM BF16 are 3.13, 3.85, and 4.69 points.

This isolates the severe regression to the current vLLM Marlin MoE path for
this Qwen3.5 MoE / compressed-tensors W4A16 / SM120 / TP4 combination. Dense
W4A16 Linear layers remain on `MarlinLinearKernel`; only MoE experts are forced
to `TritonWNA16Experts` with `--moe-backend triton`.

## Rejected or diagnostic paths

- Ordinary Transformers loading is not a valid checkpoint oracle here. It
  constructs fused expert parameters before llmcompressor can linearize the
  model, so the saved per-expert compressed keys are unexpected and fused
  expert parameters are missing.
- SGLang BF16 TP4 with the original conditional-generation checkpoint failed
  in multimodal CUDA IPC because peer access is not supported between the
  selected PCIe GPUs. This does not invalidate the SGLang W4 text-only run.
- `--language-only` was removed from the monolithic SGLang Gate. In this
  SGLang revision it enables encoder-disaggregation semantics; at TP4 the
  Qwen3.5 text class is rejected on that path.

## Evidence directories

- BF16 vLLM: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/bf16_vllm_tp4`
- W4 vLLM Marlin: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/w4a16_vllm_tp4`
- W4 vLLM Triton: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/w4a16_vllm_triton_tp4`
- W4 SGLang: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/w4a16_sglang_tp4_retry1`
- Comparisons: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/compare_bf16_vs_w4_{marlin,triton}.json`

## Gate decision

Proceed to the formal frozen quality set with vLLM BF16 as reference and vLLM
W4A16 using `VLLM_MOE_BACKEND=triton`. Keep vLLM Marlin as a rejected backend
candidate and retain SGLang W4 as independent checkpoint-quality evidence.
