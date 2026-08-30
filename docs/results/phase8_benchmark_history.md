# 阶段 8：正式矩阵与校准输入

本页说明 Phase 8 selector 和暖态 placement 使用的正式性能输入。最终在线准入不由单次
benchmark 决定，而是继续经过质量、路由稳定性、有限 canary 和自动回滚闭环；当前结论见
[暖态 placement 最终验收](phase8_warm_placement_final_acceptance_20260825.md)。

## 1. 四候选五次重复

保留的四个候选按 seed=42 随机顺序启动，每个候选重复 5 次；每次服务执行冻结的 12-cell
workload matrix。共得到 240 个 candidate/repetition/workload summary，所有请求均完成，
`failed=0`，输入 token 数、chat-template 哈希和实际 EP rank 均通过一致性检查。

10,000 次 bootstrap 聚合结果保存在
[`Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.json`](../Q-TopoMoE_Phase8_all_formats_repeated_bootstrap_20260810.json)。
原始请求、GPU 采样、元数据和日志位于服务器
`/data/models/test/qtopomoe_phase8_repeated_v4/`。

| 候选 | GPU | EP rank | median e2e p99（ms） | median 输出 tok/s | oracle 胜出数 | 峰值显存和（MiB） |
|---|---:|---:|---:|---:|---:|---:|
| FP8 TP2 PIX 0,1 Triton | 2 | 0 | 1135.29 | 828.65 | 7 | 58,878 |
| W4A16 static EP4 Triton | 4 | 4 | 1065.03 | 874.54 | 1 | 119,652 |
| RedHatAI NVFP4 static EP4 NUMA0 | 4 | 4 | 1481.23 | 918.60 | 0 | 117,540 |
| RedHatAI NVFP4 static EP8 SYS | 8 | 8 | 1249.63 | 1031.66 | 4 | 235,896 |

四个候选的 cell 变异系数分别为 0.86%、1.04%、5.55% 和 18.02%。因此 selector 使用重复
聚合和区间信息，不使用单次最快结果。

## 2. 缓存与到达模式校准

校准 workload 显式记录 prefix identity、cached tokens 和 inter-arrival。冻结 vLLM 的 cache page
按 1,056 token 对齐；对 4,224-token prompt，语义共享前缀为 100% 时，可实现的最大缓存比例为
75%，即 `floor((4224 - 1) / 1056) * 1056 / 4224`。

| 语义共享前缀 | 可实现缓存比例 | 实测比例 |
|---:|---:|---:|
| 0% | 0% | 0% |
| 50% | 50% | 50% |
| 100% | 75% | 75% |

Poisson 与 burst 场景分别调度，stream seed 唯一。所有候选使用同一 offered load：四候选容量预跑
最小观测 req/s × 0.70，并向下保留 6 位小数，避免为某个候选提供更容易的负载。

## 3. 108-cell 正式矩阵

正式矩阵由 12 个基础 cell、3 个前缀比例和 3 种到达模式组成，共 108 个 cell。四候选均重复
5 次，聚合结果包含 median e2e-p99、样本数和 10,000 次 bootstrap 95% 区间。

W4A16 缺失 M-bucket 通过 packed-int4 Triton WNA16 路径补测，并经过反量化数值检查，最大
绝对误差为 `7.6195e-06`。最终要求的 M 值为
`1,4,8,16,32,128,256,2048,8192,16384`。

关键机器产物：

- [`Q-TopoMoE_Phase8_formal_controlled_combined_20260812.json`](../Q-TopoMoE_Phase8_formal_controlled_combined_20260812.json)：四候选、108-cell、五重复聚合；
- [`Q-TopoMoE_Phase8_formal_controlled_merge_manifest_20260812.json`](../Q-TopoMoE_Phase8_formal_controlled_merge_manifest_20260812.json)：分拆运行来源与合并哈希；
- [`phase8_observations_formal_controlled_v1.json`](../../configs/strategies/phase8_observations_formal_controlled_v1.json)：selector 使用的结构化观测。

## 4. 与最终候选的关系

正式矩阵用于定义候选性能边界和训练 selector。最终发布组合还绑定：

- 冻结 selector 与当前 12% p95 regret 运行策略；
- 暖态负载生成的 `warm_swap_008_slots_per_layer_v1`；
- NVFP4 权重及辅助尺度迁移补丁；
- 116 题质量 A/B/A、当前 1% 路由稳定性策略、384 请求 canary；
- 20 窗口、2,560 请求的 trigger/cooldown/apply/rollback 验收。

完整关系由 [Release manifest](../Q-TopoMoE_release_manifest_20260825.json)冻结。
