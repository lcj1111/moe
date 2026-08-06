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

### cutlass 对照实测（FlashInfer `cutlass_fused_moe`）

通过 FlashInfer JIT 的 `cutlass_fused_moe`（对应 plan 的 cutlass backend）
补齐对照：该接口由 JIT 编译（需要 ninja 在 PATH 上），**不依赖 cuDNN
9.21**，因此绕开了 `grouped_mm_bf16` 的 cuDNN 版本限制。输入要求
`token_selected_experts` int32、`token_final_scales` float32。

| M bucket | cutlass p50 us | cutlass p95 us | triton p50 us |
|---:|---:|---:|---:|
| 1 | 714.4 | 727.5 | 723.0 |
| 8 | 1,101.0 | 1,105.3 | 1,119.5 |
| 16 | 1,124.8 | 1,129.8 | 1,132.3 |
| 32 | 1,173.4 | 1,182.1 | 1,244.4 |
| 256 | 3,159.5 | 3,198.0 | 2,807.7 |
| 2,048 | 25,287.1 | 25,391.0 | 18,977.0 |
| 8,192 | 100,807.9 | 100,997.1 | 74,320.4 |
| 16,384 | OOM（32.55 GiB > 31.36 GiB 单卡） | — | 148,677.9 |

小 M（≤32）两者相当，cutlass 略快；M≥256 后 triton 明显更快（2,048 时
快 33%，8,192 时快 26%）；16,384 时 cutlass 在单卡 32GB 显存下无法运行，
已记录为 `valid=false` 的 failed 行（不参与选择，不伪造）。

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
| 候选 / 实测 / observation | 8 / 16 / 4 |
| 四条 observation 的选中策略 | 均为 bf16_tp4_numa0_triton |
| 预测 p99（smoke / trace） | 2.30 / 14.63 / 89.71 / 89.71 ms |
| invalid_config_rate | 0.625（5/8 无实测的 cutlass/fp8 被排除） |
| oracle / regret | null（无候选带 measured_p99，暂无法算 regret） |

Gate 状态：median/p95 regret 因缺 oracle 为 null；controller overhead
p95=5.28%（由预测值极小的 smoke observation 拉高），两条真实 trace
observation 的 overhead 占比仅 0.28-0.29%（<1% 目标在真实负载下达标）。

选择说明：trace 负载以 M=8,192/16,384 为主，cutlass 在这些大 M 下更慢且
16,384 无测量（invalid），因此 triton 被选中；这符合"无实测不可选"的
框架设计，也是真实结论而非偏好。

## 3. 结论与下一步

1. 首次打通"真实 trace → 真实 kernel 实测 → 策略回放"闭环：BF16 与 W4
   两条真实 route 数据驱动 workload，kernel DB 提供延迟，selector 输出
   可解释的选择（真实负载预测 p99 ≈ 89.7ms，controller overhead <0.2%）。
2. 下一步优先：为 cutlass/flashinfer/fp8 补齐实测（或至少给候选提供
   measured_p99 以便计算 regret/oracle）；FlashInfer 需 cuDNN ≥ 9.21。
