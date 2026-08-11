# Qwen3.5 MoE 原始检查点上游 Gate（2026-08-05）

## 结果

冻结的源检查点在两个 clean-room 上游框架中均被拒绝；测试未使用 adapter、`PYTHONPATH` 注入或模型类覆盖。两次失败都指向同一个导出命名空间不匹配问题。

| 后端 | 精确版本 | TP/GPU | 结果 | 首个可操作原因 |
|---|---|---|---|---|
| vLLM | `0.26.1rc1.dev343+g33c50587d` | TP1/GPU 0 | rejected | 原生 `Qwen3_5Model` 没有 `language_model` 子模块 |
| SGLang | `0.5.17.dev0+g6c05aaae7e` | TP1/GPU 1 | rejected | `language_model.*` 键与 text-only loader 不匹配 |

vLLM 已构造原生 Qwen3.5 MoE 模型并进入正常权重 loader，随后报错：

```text
ValueError: There is no module or parameter named 'language_model' in Qwen3_5Model.
```

SGLang 选择上游 `qwen3_5_text.py` 与 `CompressedTensorsWNA16MarlinMoEMethod`，随后记录缺失的 `language_model.*` 参数并报错：

```text
KeyError: 'language_model.layers.0.mlp.experts.w2_weight_packed'
```

证据目录：

- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_original/vllm_33c505_tp1`
- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_original/sglang_6c05_tp1`

## 决策

配置已经是 text-only（`Qwen3_5MoeForCausalLM`、`qwen3_5_moe_text`），compressed-tensors 的 ignore path 也已经使用 `model.layers.*`；只有权重文件仍保留 VLM wrapper 命名空间。因此下一产物应为独立检查点，只做以下确定性键转换：

```text
model.language_model.* -> model.*
```

`lm_head.weight`、config/tokenizer 文件、张量 shape、dtype 和 value 保持不变；绝不覆盖冻结的源目录。
