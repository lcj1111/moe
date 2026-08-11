# 阶段 4 / 阶段 8 可执行框架

本框架实现 CPU 侧控制平面，不宣称 CUDA benchmark 已完成。只有 kernel 数据行同时满足 `measured: true`、`valid: true` 且含 p50 或 p95 延迟时，selector 才允许默认选中；因此在收集 CUTLASS/Triton/FlashInfer 实测前，提交空 kernel DB 模板不会产生误选。

官方 RedHatAI NVFP4 几何参数已经在接受的 trace bucket 中加入 `vllm.run_cutlass_moe_fp4` 数据行。50 次重复的 p95 延迟为 M=1：250.42 us，M=2048：728.23 us，M=8192：2219.58 us；原始数据为 `docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_real_m_20260809.json`。这些 CUTLASS 直接测量不能替代 EP 服务数据，因为冻结 vLLM 在专家分片服务中选择了 MARLIN。

Runbook 还要求 7 个 NVFP4 bucket。相同 RTX 5090/CUTLASS 路径的 50 次重复 p50/p95（微秒）为：M=4 235.61/253.71，M=8 232.72/240.26，M=16 285.39/293.39，M=32 361.28/369.05，M=128 453.48/461.00，M=256 475.03/485.84，M=16384 4382.33/4413.00。与 M=1/2048/8192 合并后，覆盖冻结的 12-cell workload matrix；原始数据为 `docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_phase8_missing_m_20260809.json`。

## M-bucket workload 生成

```bash
python3 phase4/workload/generate_m_buckets.py \
  --output configs/workloads/m_buckets.json --seed 42
```

manifest 有 108 个确定性 case：W1（256/128，C=1/8/32/128）、W2（2048/256，C=1/8/32）、W3（8192/256，C=1/8/16）、W4（32768/128，C=1/4），并与 prefix-cache 0/50/100% 和 closed-loop/Poisson/burst 到达方式交叉。`prefill_m=input_tokens*concurrency`，`decode_m=concurrency`，二者均向上取整到配置的二次幂 bucket；`real_M_hist` 保留 Phase 8 成本模型需要的 prefill/decode 混合分布。

## 阶段 4 kernel / backend 选择器

在 `configs/kernels/phase4_kernel_db.json` 填入实测数据行，例如 `backend=cutlass`、`kernel_config={tile_m,tile_n,tile_k,stages,warps}`、`m_bucket`、`precision`、`p50_us`、`p95_us`、`measured=true` 和 source run ID。随后加载 `KernelDatabase` 并调用 `BackendSelector.select`。缺失或无效数据会抛出 `SelectionError`；`allow_unmeasured=True` 只能作为开发期显式绕过，不能用于正式结论。

## 阶段 8 策略选择器

`StrategyCandidate` 包含 Runbook 字段：量化格式/checkpoint、TP/DP/EP、GPU 映射、EPLB 策略、冗余 expert、kernel backend 和 kernel config。`StrategySelector` 依次排除不支持、质量不合格和显存超限的候选。`CostModel` 计算：

`compute(real_M_hist, kernel_db) + communication_bytes * measured mapping cost + imbalance(route_hist) + migration_bytes * migration cost`

通信映射速率由 `CostModel.communication_us_per_gb_by_mapping` 提供；默认速率只允许作为测量前 dry-run 的记录性 fallback。`evaluate()` 输出 top-1、median/p95 regret、决策开销占预测 p99 的百分比和无效配置率。Runbook Gate 为 median regret ≤5%、p95 ≤10%、controller overhead <1%，以显式字段输出；透明模型未证明失败前不引入 RL。

## 运行时约束

- 所有服务必须使用冻结的环境 pin、chat-template hash 和 P2P 设置；不可在 benchmark 中偷偷切换 backend 或 vLLM 版本。
- 机器可读结果必须记录输入 manifest、运行 seed、实际 EP rank 和服务日志路径。
- 未测量、缺失或由 proxy 推断的延迟必须标记为不可接受，不能伪装成真实 kernel 数据。
- 详细的阶段 8 筛选、重复运行、容量冻结和校准结果见[阶段 8 基准与校准历史](results/phase8_benchmark_history.md)。
