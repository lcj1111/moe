# W4A16 real-load Gate (2026-08-05)

Static coverage passed: the checkpoint contains 40 layers × 256 experts × 3
expert projections = 30,720 packed expert linears and 30,720 scales, with
`linear_attn` excluded. The checkpoint is `compressed-tensors` W4A16 and is
20G on disk.

The real service Gate did not pass. vLLM 0.26.0 failed before allocation with a
Qwen3.5 MoE config type mismatch (`Qwen3_5MoeTextConfig` versus the vLLM
`Qwen3_5MoeConfig` expected by its multimodal renderer). SGLang 0.5.16 failed
before serving because it has no compatible `Qwen3_5MoeForCausalLM`
implementation. These are loader/architecture failures, not OOM evidence.

Consequently `/health`, model discovery, completion, metrics, concurrency
smoke, quality comparison and route capture are not yet valid for W4A16. TP,
EP, EPLB and Phase 8 real strategy selection remain blocked until a supported
loader or an explicitly reviewed model-adapter fix is validated.
