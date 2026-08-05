# Qwen3.5 MoE original-checkpoint upstream Gate (2026-08-05)

## Result

The frozen source checkpoint was rejected by both clean-room upstream
frameworks without adapters, `PYTHONPATH` injection, or model-class overrides.
Both failures identify the same export namespace mismatch.

| Backend | Exact version | TP/GPU | Result | First actionable cause |
|---|---|---|---|---|
| vLLM | `0.26.1rc1.dev343+g33c50587d` | TP1/GPU 0 | rejected | Native `Qwen3_5Model` has no `language_model` child |
| SGLang | `0.5.17.dev0+g6c05aaae7e` | TP1/GPU 1 | rejected | `language_model.*` keys do not match the text-only loader |

vLLM constructed the native Qwen3.5 MoE model and reached its normal weight
loader, then raised:

```text
ValueError: There is no module or parameter named 'language_model' in Qwen3_5Model.
```

SGLang selected its upstream `qwen3_5_text.py` implementation and
`CompressedTensorsWNA16MarlinMoEMethod`, then logged missing
`language_model.*` parameters and raised:

```text
KeyError: 'language_model.layers.0.mlp.experts.w2_weight_packed'
```

Evidence:

- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_original/vllm_33c505_tp1`
- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_original/sglang_6c05_tp1`

## Decision

The config is already text-only (`Qwen3_5MoeForCausalLM`,
`qwen3_5_moe_text`), and its compressed-tensors ignore paths already use
`model.layers.*`. Only the weight file retains the VLM wrapper namespace.

The next artifact will therefore be a separate checkpoint with this one key
mapping:

```text
model.language_model.* -> model.*
```

`lm_head.weight`, config/tokenizer files, tensor shapes, dtypes, and tensor
values remain unchanged. The frozen source directory is never overwritten.
