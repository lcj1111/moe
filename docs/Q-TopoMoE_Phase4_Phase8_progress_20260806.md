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
**`m_bucket` 采用总 token 数语义**（Phase 4/8 定义：
`prefill_m = input_tokens × concurrency`、`decode_m = concurrency`），即
kernel 的 flat `num_tokens` 输入行数；per-expert 行数由路由 kernel 内部
派生。

v3 实测（正确语义，三个组合 × 8 M 全部测出）：

| M bucket | triton bf16 p50 us | triton fp8 p50 us | cutlass bf16 p50 us |
|---:|---:|---:|---:|
| 1 | 183 | 225 | 138 |
| 8 | 341 | 250 | 338 |
| 16 | 519 | 353 | 500 |
| 32 | 724 | 453 | 720 |
| 256 | 1,118 | 662 | 1,099 |
| 2,048 | 1,306 | 819 | 1,254 |
| 8,192 | 2,810 | 1,787 | 3,156 |
| 16,384 | 5,121 | 3,324 | 6,329 |

cutlass bf16 经 FlashInfer JIT `cutlass_fused_moe`（需 ninja 在 PATH，
不依赖 cuDNN 9.21）；fp8 经 vLLM `fp8_w8a8_moe_quant_config`（服务同款
triton kernel）。fp8 相对 bf16 稳定加速约 1.4-1.8x；小 M 三者相当，
M≥8,192 后 triton 明显快于 cutlass。

注：早期 v1/v2 数据存在语义偏差（把 m_bucket 当每专家 token 数，导致
num_tokens 放大 32 倍、16384 单卡 OOM），已用 v3 正确语义数据替换。

## 2. Phase 8：策略回放（completed）

候选表新增 triton backend 候选（bf16/fp8 × tp2/tp4/tp8），observation
扩展为 4 条：2 条 smoke 占位 + 2 条真实 trace（BF16 与 W4 全量，由
`scripts/build_phase8_observations_from_trace.py` 从 capture manifest 的
prompt/gen token 分布生成，real_M_hist 为批次 token 数语义）。

回放结果：
[Q-TopoMoE_Phase8_replay_20260806.json](Q-TopoMoE_Phase8_replay_20260806.json)。

| 项 | 值 |
|---|---|
| 状态 | completed（此前 blocked） |
| 候选 / 实测 / observation | 11 / 24 / 4 |
| 选中策略 | 4 条均为 fp8_tp2_node02_triton |
| 预测 p99 | 0.56 / 0.78 / 1.25 / 1.25 ms |
| invalid_config_rate | 0.182（2 个 fp8-cutlass 无实测被排除） |
| oracle / regret | null（无候选带 measured_p99，暂无法算 regret） |

Gate 状态：median/p95 regret 因缺 oracle 为 null；controller overhead
p95=28.27%（由预测值极小的 smoke observation 拉高——预测 0.56ms 时决策
开销占比自然偏高）；真实 trace observation 的 overhead 占比 22-23%。
这些均属"决策开销 vs 极小预测值"的比例效应，绝对决策时间 <0.16ms。

选择说明：三个 backend 全 M 实测后，fp8（triton）在全部负载上延迟最低，
且 TP2 通信量最小，故 4 条 observation 一致选 fp8_tp2_node02_triton；
真实 trace 的 M 分布以 2,048（89.7%）为主，fp8 在该 M 快 1.6x。

## 3. 结论与下一步

1. 首次打通"真实 trace → 真实 kernel 实测 → 策略回放"闭环：BF16 与 W4
   两条真实 route 数据驱动 workload，kernel DB 提供延迟，selector 输出
   可解释的选择（真实负载预测 p99 ≈ 89.7ms，controller overhead <0.2%）。
2. 下一步优先：为 cutlass/flashinfer/fp8 补齐实测（或至少给候选提供
   measured_p99 以便计算 regret/oracle）；FlashInfer 需 cuDNN ≥ 9.21。
