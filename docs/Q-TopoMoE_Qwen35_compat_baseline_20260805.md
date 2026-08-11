# Qwen3.5 MoE W4A16 兼容性基线（2026-08-05）

## 冻结检查点

- 路径：`/data/models/test/qtopomoe_w4a16`
- 架构：`Qwen3_5MoeForCausalLM`
- model type：`qwen3_5_moe_text`
- 量化：compressed-tensors W4A16
- 张量数量：93,093
- `model.safetensors`：20,928,221,896 bytes
- `model.safetensors` SHA-256：`36e5b5ec77c5c35e3ce23f415e31c7fcbd5b14e287f02629ce057698cdd6d94f`
- 八文件完整 manifest：`docs/Q-TopoMoE_W4A16_freeze_20260805.json`

该检查点具有混合导出身份：配置是 text-only，但 93,092/93,093 个张量名仍位于 `model.language_model.*` 下，唯一的根级张量是 `lm_head.weight`。这是导出/布局兼容性问题，不能据此判断 GPTQ/W4A16 算术无效。

## 冻结的服务环境

| 后端 | 框架 | Transformers | compressed-tensors | Torch | FlashInfer |
|---|---:|---:|---:|---:|---:|
| vLLM | 0.26.0 | 5.14.1 | 0.17.0 | 2.11.0 | 0.6.14 |
| SGLang | 0.5.16 | 5.12.1 | 0.17.2a20260728 | 2.11.0 | 0.6.14 |

两个上游 clean-room 环境也已为下一 Gate 冻结：

- vLLM 官方 nightly 精确提交：`33c50587d2679ba9bacc2a51ae19901f7eb3a129`（`0.26.1rc1.dev343+g33c50587d`，Torch `2.13.0+cu130`）；
- SGLang 源码提交：`6c05aaae7e3966469b6c552aa11b545e5d27f8bf`（`0.5.17.dev0+g6c05aaae7e`，`sglang-kernel==0.4.5`，FlashInfer `0.6.15.post1`，Torch `2.11.0+cu130`）；
- Transformers 参考提交：`d24d79da55f7ee6e538a460d3025e41dcc41ab21`。

最初记录的 vLLM 源码引用没有匹配的预编译 wheel，因此没有覆盖已发布环境。当前选择与官方 wheel index 宣称的提交一致，避免 Python 源码与 CUDA 二进制不匹配。SGLang 使用固定源码和上游支持的 `SGLANG_BUILD_RUST_EXTS=none` 构建，CUDA serving kernel 仍来自固定的打包依赖。

## 基于证据的状态

### vLLM 开发 adapter

开发阶段 adapter 通过 `/health`、`/v1/models`、真实 chat completion（`42`）和 `/metrics`，并记录了原生 compressed-tensors、Marlin linear 与 Marlin MoE 选择。这只能证明 smoke 级数值可用，不能作为最终部署结果，因为它使用了 vLLM 开发/调试类覆盖和启动器内部 config hook。

证据目录：`/data/models/test/qtopomoe_w4a16_runs/vllm_adapter_tp1_hybrid/acceptance`

### SGLang 开发 adapter

该 adapter 被拒绝：仅加载约 17.53 GiB，并对 `lm_head` 与多个 GDN projection/output 权重报 `Parameter ... not found`，随后在 CUDA Graph capture 阶段因 attention-backend 接口错误失败。不得把它当作有效加载或服务结果。

证据目录：`/data/models/test/qtopomoe_w4a16_runs/sglang_adapter_tp1_weights`

## 正确的后续策略

1. 不再向已拒绝的 adapter 添加兼容字段；
2. 先在隔离环境测试固定版本的上游框架源码；
3. 要求 checkpoint 参数零缺失、零跳过；
4. 若冻结检查点仍失败，则通过确定性的 tensor-key 转换创建独立的规范化服务检查点，绝不覆盖冻结源；
5. 只有在 health、发现、真实 completion、metrics、参数覆盖和 BF16 质量对照全部通过后，才接受某个后端。
