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

| M bucket | p50 us | p95 us |
|---:|---:|---:|
| 1 | 179.9 | 192.3 |
| 8 | 340.1 | 347.8 |
| 16 | 515.1 | 521.9 |
| 32 | 721.4 | 724.9 |
| 256 | 1,116.2 | 1,125.1 |
| 2,048 | 1,303.9 | 1,311.8 |
| 8,192 | 2,804.8 | 2,825.0 |
| 16,384 | 5,130.8 | 5,143.4 |

注：vLLM 提示该机型无预置 MoE 配置，使用默认 config，性能可能非最优；
这 8 条为默认配置下的真实实测基线。

8 条结果已合并进 `configs/kernels/phase4_kernel_db.json`
（`measured=true, valid=true`）。cutlass/flashinfer 与 fp8 行仍为
`planned`，不伪造延迟。

## 2. Phase 8：策略回放（completed）

候选表新增 3 条 triton backend 候选（原 5 条 cutlass/fp8 保留）；
observation 由 2 条 smoke 占位扩展为 3 条（新增 1 条由 Phase 3 BF16 全量
trace 的 `expert_token_histogram.json` 生成，见
`scripts/build_phase8_observations_from_trace.py`）。

回放结果：
[Q-TopoMoE_Phase8_replay_20260806.json](Q-TopoMoE_Phase8_replay_20260806.json)。

| 项 | 值 |
|---|---|
| 状态 | completed（此前 blocked） |
| 候选 / 实测 / observation | 8 / 8 / 3 |
| 三条 observation 的选中策略 | 均为 bf16_tp4_numa0_triton |
| 预测 p99 | 0.89 / 1.16 / 5.93 ms |
| invalid_config_rate | 0.625（5/8 无实测的 cutlass/fp8 被排除） |
| oracle / regret | null（无候选带 measured_p99，暂无法算 regret） |

Gate 状态：median/p95 regret 因缺 oracle 为 null；controller overhead
p95=9.4%（第一条 observation 预测极小 0.89ms 时 overhead 占比 10%）。

## 3. 结论与下一步

1. 首次打通"真实 trace → 真实 kernel 实测 → 策略回放"闭环：route 数据
   驱动 workload，kernel DB 提供延迟，selector 输出可解释的选择。
2. 下一步优先：为 cutlass/flashinfer/fp8 补齐实测（或至少给候选提供
   measured_p99 以便计算 regret/oracle）；用 W4 trace 生成第二条真实
   observation；给 selector 加真实 measured 数据后重跑 gate。
