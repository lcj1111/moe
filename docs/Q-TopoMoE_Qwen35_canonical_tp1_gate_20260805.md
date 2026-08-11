# Qwen3.5 MoE 规范化 W4A16 TP1 Gate（2026-08-05）

值完全一致的规范化检查点在原生 vLLM 与 SGLang TP1 服务中通过，未使用 adapter、`PYTHONPATH` 注入或模型类覆盖。

| 后端 | 原生量化路径 | 权重加载 | GPU 权重显存 | health/models/completion/metrics | 覆盖警告 |
|---|---|---:|---:|---|---:|
| vLLM | Marlin linear + Marlin WNA16 MoE | 21.40 s | 19.53 GiB | 通过（`42`） | 0 bytes |
| SGLang | CompressedTensors WNA16 Marlin MoE | 19.00 s | 19.75 GiB | 通过（`42`） | 0 bytes |

## 检查点

- 路径：`/data/models/test/qtopomoe_w4a16_canonical_text_v1`
- 权重 SHA-256：`0eb2775989321d038ea9534041a6030f536a0d0d2293e7087837390e9738c1d2`
- 张量一致性：93,093/93,093；已对 20,915,187,456 个张量字节验证 shape、dtype 和完整 value 一致；
- 配置是否改变：否。

## 证据目录

- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_canonical/vllm_33c505_tp1_retry1`
- `/data/models/test/qtopomoe_w4a16_runs/cleanroom_canonical/sglang_6c05_tp1`

该结果只关闭了架构和权重布局兼容性 Gate，尚未关闭 BF16 质量对照、重复性能基准、TP2/TP4/TP8 或 CUDA Graph 生产模式验证。
