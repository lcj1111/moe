# 阶段 7–8：在线迁移与四候选正式选择器收尾

> 更新日期：2026-08-12（Asia/Shanghai）  
> 结论：阶段 7 在线 placement-plan/迁移 Gate 已接受；阶段 8 的四候选正式测量已接受，但选择器 p95 regret 未达标，尚不可在线启用。

## 1. 阶段 7 在线 placement-plan 与迁移

NVFP4 EP8 服务通过显式启用的运行时 bridge 应用冻结 placement-plan。实验覆盖稳定窗口、在线迁移阻塞窗口和恢复窗口，共 384 个请求，失败数为 0。

| 指标 | 结果 |
|---|---:|
| 稳定窗口端到端 p99 | 1082.04 ms |
| 在线迁移窗口端到端 p99 | 2166.96 ms |
| 恢复窗口端到端 p99 | 988.83 ms |
| 首次原生 rearrangement | 0.83 s |
| 运行时 plan SHA-256 | `d7d7240007c8d69cce9350ff6e4deca250713282c651d32c954b077b383763fd` |

详细过程见[在线 placement-plan 与服务迁移实验](../Q-TopoMoE_在线placement-plan与服务迁移实验_20260812.md)，专家迁移微基准见[阶段 7 迁移微基准数据](../Q-TopoMoE_Phase7_migration_microbench_20260812.json)。

原生 vLLM 对 `CompressedTensorsW4A4Nvfp4MoEMethod` 的 EPLB 限制仍保留为历史事实。本次接受的是显式 opt-in bridge Gate，不能改写为“原生 vLLM EPLB 已支持 NVFP4”。

## 2. 阶段 8 四候选正式测量

两组已完成实验被无损合并：GPU0–3 目录提供 FP8 TP2、NVFP4 EP4、W4A16 EP4 共 15 次运行；EP8 目录提供 NVFP4 EP8 共 5 次运行。合并目录只创建指向原始 run 的符号链接，不复制或修改原始结果。

- workload：108 个确定性 cell；
- 每个候选、每个 cell：5 次重复；
- bootstrap：10,000 次；
- Gate：运行状态、cell 完整性、失败请求、请求完成度、token 精确性、prefix-cache、到达调度、EP rank 与 chat template 全部通过；
- 合并聚合状态：`accepted`，错误列表为空。

| 候选 | GPU 数 | cell 中位 p99 的中位数 | 输出吞吐中位数 | Oracle 胜场 |
|---|---:|---:|---:|---:|
| FP8 TP2 | 2 | 1610.56 ms | 148.17 token/s | 46 |
| NVFP4 EP4 | 4 | 1496.40 ms | 141.92 token/s | 12 |
| NVFP4 EP8 | 8 | 1403.87 ms | 140.78 token/s | 45 |
| W4A16 EP4 | 4 | 1729.80 ms | 124.99 token/s | 5 |

资源感知 Pareto 保留 FP8 TP2、NVFP4 EP4 和 NVFP4 EP8；W4A16 EP4 被支配。完整聚合见[四候选正式聚合数据](../Q-TopoMoE_Phase8_formal_controlled_combined_20260812.json)，来源与 schedule 哈希见[合并清单](../Q-TopoMoE_Phase8_formal_controlled_merge_manifest_20260812.json)。

## 3. 选择器 Gate

旧透明成本模型只表示 M 桶、通信和路由不均衡，不能表示正式矩阵新增的 prefix-cache 与 Poisson/burst 排队。按 Runbook 的失败处理要求，改用更简单的单近邻选择器做误差审计：

1. 严格匹配 `prefix_cache_pct` 和 `arrival_mode`；
2. 对输入 token、输出 token、并发和冻结请求率计算对数距离；
3. 每轮整体留出 W1/W2/W3/W4 中的一族，查找和选择时不读取留出族的测量或 oracle。

| 指标 | 结果 | Gate |
|---|---:|---|
| top-1 | 56.48% | 仅报告 |
| median regret | 0.00% | 通过（≤5%） |
| p95 regret | 43.17% | **失败**（要求≤10%） |
| 控制开销 p95 / 服务 p99 | 0.0107% | 通过（<1%） |
| 不可行配置误选率 | 0 | 通过 |

108 个 cell 中只有 50 个的 oracle 与其他候选 bootstrap 95% 区间完全分离，表明候选排序既有真实 workload 切换，也存在重复测量波动。无论原因如何，正式 Gate 必须按全体 cell 计算，因此当前结论仍是 `gate_failed`。

机器可读结果见[简化选择器正式 Gate](../Q-TopoMoE_Phase8_formal_selector_nearest_gate_20260812.json)。当前不得执行依赖该 selector 的动态 trigger/cooldown/rollback 在线闭环，也不得把该结果描述为生产可用。

## 4. 后续顺序

1. 增加独立的服务状态特征：实际 KV-cache 命中、排队深度、近期到达率与短窗口 p99；
2. 冻结模型结构与阈值后，新增独立测试 workload，而不是继续在现有 108-cell 上调参；
3. 仅当 median/p95 regret、控制开销和不可行误选率全部通过后，再执行动态 trigger、cooldown 与 rollback Gate；
4. 与 selector 解耦的 full-set FP8/NVFP4 质量收尾可独立排期。

## 5. 复现入口

```bash
python3 scripts/combine_phase8_split_runs.py \
  --source-root /data/models/test/qtopomoe_phase8_formal_controlled_gpu0_3_v1 \
  --source-root /data/models/test/qtopomoe_phase8_formal_controlled_ep8_v1 \
  --output-root /data/models/test/qtopomoe_phase8_formal_controlled_combined_v1 \
  --workload-matrix /data/moe/configs/workloads/phase8_formal_controlled_v1.json

python3 scripts/aggregate_phase8_repeated.py \
  --run-root /data/models/test/qtopomoe_phase8_formal_controlled_combined_v1 \
  --output /data/models/test/qtopomoe_phase8_formal_controlled_combined_v1/aggregate.json \
  --bootstrap-samples 10000

python3 scripts/evaluate_phase8_nearest_selector.py \
  --aggregate docs/Q-TopoMoE_Phase8_formal_controlled_combined_20260812.json \
  --output docs/Q-TopoMoE_Phase8_formal_selector_nearest_gate_20260812.json
```
