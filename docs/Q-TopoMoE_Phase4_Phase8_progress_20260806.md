# Q-TopoMoE Phase 4/8 进展：Triton MoE kernel 实测与策略回放

> 生成日期：2026-08-06（Asia/Shanghai）
> 摘要：Phase 4 kernel DB 从空模板首次获得 8 条真实 Triton MoE kernel
> 实测（vLLM `fused_experts`，服务同款）；Phase 8 回放从
> `blocked_missing_kernel_measurements` 转为 `completed`，并由真实 Phase 3
> trace 生成 workload observation 驱动。

## 1. Phase 4：Triton MoE kernel 实测

基准脚本：`scripts/bench_moe_kernel.py`；原始数据：
[Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806.json](Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806.json)。

实测对象为 vLLM cleanroom `fused_experts`（Triton MoE kernel），与服务
实际执行路径一致，几何参数取 Qwen3.6-35B-A3B text config
（hidden=2048、moe_intermediate=512、experts=256、top_k=8）。
`m_bucket` 采用**每专家 token 数**语义（与 Phase 8 `real_M_hist` 一致）：
triton 调用以 `num_tokens = m_bucket * num_experts // top_k` 输入，使每个
专家恰好收到 `m_bucket` 行。

修正后实测（v2，per-expert 语义）：

| M bucket | p50 us | p95 us |
|---:|---:|---:|
| 1 | 723.0 | 732.6 |
| 8 | 1,119.5 | 1,126.2 |
| 16 | 1,132.3 | 1,140.2 |
| 32 | 1,244.4 | 1,255.2 |
| 256 | 2,807.7 | 2,815.0 |
| 2,048 | 18,977.0 | 19,014.2 |
| 8,192 | 74,320.4 | 74,838.1 |
| 16,384 | 148,677.9 | 149,167.6 |

注：vLLM 提示该机型无预置 MoE 配置，使用默认 config，性能可能非最优；
这 8 条为默认配置下的真实实测基线。

8 条结果已合并进 `configs/kernels/phase4_kernel_db.json`
（`measured=true, valid=true`）。

**FlashInfer 对照未完成（环境限制，如实记录）**：`grouped_mm_bf16` 仅支持
cudnn backend 且要求 cuDNN ≥ 9.21，本机为 9.20，实测报
`cuDNN grouped_mm_bf16 requires backend version >= 92100, found 92000`。
因此 flashinfer/fp8/cutlass 行保持 `planned`，不伪造延迟；后续升级 cuDNN
或提供对应 kernel 后补齐。

## 2. Phase 8：策略回放（completed）

候选表新增 3 条 triton backend 候选（原 5 条 cutlass/fp8 保留）；
observation 扩展为 4 条：2 条 smoke 占位 + 2 条真实 trace（BF16 与 W4
全量 `expert_token_histogram.json`，见
`scripts/build_phase8_observations_from_trace.py`）。

回放结果：
[Q-TopoMoE_Phase8_replay_20260806.json](Q-TopoMoE_Phase8_replay_20260806.json)。

| 项 | 值 |
|---|---|
| 状态 | completed（此前 blocked） |
| 候选 / 实测 / observation | 8 / 8 / 4 |
| 四条 observation 的选中策略 | 均为 bf16_tp4_numa0_triton |
| 预测 p99（smoke / trace） | 2.30 / 14.63 / 89.71 / 89.71 ms |
| invalid_config_rate | 0.625（5/8 无实测的 cutlass/fp8 被排除） |
| oracle / regret | null（无候选带 measured_p99，暂无法算 regret） |

Gate 状态：median/p95 regret 因缺 oracle 为 null；controller overhead
p95=3.3%，两条真实 trace observation 的 overhead 占比仅 0.17-0.19%
（<1% 目标在真实负载下达标）。

## 3. 结论与下一步

1. 首次打通"真实 trace → 真实 kernel 实测 → 策略回放"闭环：BF16 与 W4
   两条真实 route 数据驱动 workload，kernel DB 提供延迟，selector 输出
   可解释的选择（真实负载预测 p99 ≈ 89.7ms，controller overhead <0.2%）。
2. 下一步优先：为 cutlass/flashinfer/fp8 补齐实测（或至少给候选提供
   measured_p99 以便计算 regret/oracle）；FlashInfer 需 cuDNN ≥ 9.21。
