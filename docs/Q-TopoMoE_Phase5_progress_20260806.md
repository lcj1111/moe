# Q-TopoMoE Phase 5：Level 2 融合 kernel（permute + quant/scale + pack）

> 生成日期：2026-08-06（Asia/Shanghai）
> 目标：按 Runbook Phase 5 选择 `permute + activation quant/scale + pack`
> 融合作为 Level 2 首个目标；先 profile 占比，再实现 Triton 融合 kernel，
> 做极端 shape 正确性与 micro 收益验证。

## 1. Profile：目标阶段占比

在真实几何（Qwen3.6-35B-A3B：hidden 2048、moe_intermediate 512、
experts 256、top_k 8、M=2048）上，用服务同款组件实测：

| 阶段 | 耗时 | 占 fused MoE |
|---|---:|---:|
| permute（vLLM `moe_permute`） | 0.113 ms | 8.73% |
| activation quant（vLLM fp8 kernel） | 0.060 ms | 4.60% |
| **目标合计（permute+quant）** | 0.173 ms | **13.34%** |
| fused_experts 总时间 | 1.298 ms | 100% |

Runbook 停止阈值是目标阶段 <10%；实测 13.34% > 10%，因此值得做融合。
分析脚本：`scripts/profile_moe_stages.py`。

## 2. 融合 kernel：`scripts/fused_permute_quant.py`

单个 Triton kernel 一次遍历完成：

1. Pass 1：逐 token 计算行内 |x| max → fp8 per-token scale；
2. Pass 2：按 top-k 专家 id 写入 per-expert packed fp8(e4m3) 激活 + scale。

pre-packing 用 host 侧 `bincount` + 前缀和（与 vLLM 排序语义一致）。
正确性以 fp32 精确 scale 的 Torch reference 为基准（避免 bf16 中间舍入
噪声），fp8 反量化逐元素对比，容差 1 ulp。

## 3. 极端 shape 正确性

全部 7 个 shape 通过（max_err = 0.0000），含 decode 极值 M=1 与
prefill 极值 M=16,384：

| M | correct | max_err |
|---:|---|---:|
| 1 | true | 0.0 |
| 8 | true | 0.0 |
| 32 | true | 0.0 |
| 256 | true | 0.0 |
| 2,048 | true | 0.0 |
| 8,192 | true | 0.0 |
| 16,384 | true | 0.0 |

## 4. Micro 收益（vs Torch reference）

| M | ref ms | fused ms | speedup |
|---:|---:|---:|---:|
| 1 | 0.126 | 0.224 | 0.56x |
| 8 | 0.122 | 0.226 | 0.54x |
| 32 | 0.115 | 0.258 | 0.45x |
| 256 | 0.120 | 0.258 | 0.46x |
| 2,048 | 0.683 | 0.283 | 2.41x |
| 8,192 | 2.893 | 0.336 | 8.61x |
| 16,384 | 5.734 | 0.677 | 8.47x |

prefill 类大 M 收益 2.4-8.6x；decode 类小 M（≤256）受 Triton 启动开销
影响反而更慢（0.45-0.56x）。原始数据：
[Q-TopoMoE_Phase5_permute_quant_fusion_20260806.json](Q-TopoMoE_Phase5_permute_quant_fusion_20260806.json)。

## 5. 结论与边界

1. 目标阶段（permute+quant）占 fused MoE 13.34%，超过 Runbook 10% 停止
   阈值；相对 naive Torch reference，大 M（prefill）micro 收益 2.4-8.6x。
2. 当前实现为 standalone micro kernel（未接入 vLLM modular experts），
   满足 Runbook"Torch reference → Triton → 极端 shape 正确性"前三步。

## 6. 更新（2026-08-07）：prepare-stage A/B 结论 —— 停止 Level 2

用同一输入对比 vLLM 服务路径的 prepare 组件（`moe_permute` +
`per_token_group_quant_fp8`，Side A）与我们的融合 kernel（Side B）：

| M | A（vLLM prepare） | B（fused） | A/B |
|---:|---:|---:|---:|
| 1 | 0.087 ms | 0.239 ms | 0.37x |
| 2,048 | 0.157 ms | 0.295 ms | 0.53x |
| 8,192 | 0.579 ms | 0.348 ms | 1.67x |
| 16,384 | 1.174 ms | 0.691 ms | 1.70x |

按真实 trace 的 token 加权 M 分布（2048 占 89.7%、8192 占 9.5%、1 占
0.7%）：A=0.197ms、B=0.300ms，**B 慢 34.4%**。

**结论**：融合 kernel 只在 M≥8,192 快（1.7x），而真实负载主导的
M=2,048 下 vLLM 现有组件更快（Triton 启动开销抵消融合收益）。相对
vLLM 服务路径无端到端收益，按 Runbook"两周内关键 M 桶无 ≥10% micro
收益则停止 Level 2"的规则，**Phase 5 Level 2 停止**；融合 kernel 保留为
研究参考（`scripts/fused_permute_quant.py`），不接入 vLLM。

A/B 数据：[Q-TopoMoE_Phase5_prepare_ab_20260807.json](Q-TopoMoE_Phase5_prepare_ab_20260807.json)。
