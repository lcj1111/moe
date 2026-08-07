# Q-TopoMoE Phase 7：量化感知 EPLB（离线 placement 与迁移成本）

> 生成日期：2026-08-07（Asia/Shanghai）
> 范围：11.1 迁移成本实测 + 11.2 离线 placement 求解器与策略对比；
> 11.3 在线控制器参数框架在后续阶段接入真实服务后验证。

## 1. 迁移成本实测（11.1）

单个专家 w1+w2（BF16，2048×512×2 字节 = 4MB）的 D2D 拷贝中位耗时
（RTX 5090）：

| 路径 | 耗时/专家 |
|---|---:|
| 同 GPU（intra-device） | 10.2 μs |
| 同 NUMA（peer，GPU0→1） | 78.8 μs |
| 跨 NUMA（peer，GPU0→4） | 75.3 μs |

同 NUMA 与跨 NUMA 差异不大（PCIe/互联均为 peer 拷贝主导），说明 4 卡
场景下迁移成本主要取决于专家数量与拷贝次数，而非 NUMA 距离。
测量脚本：`phase7/migration_cost.py`。

## 2. 离线 placement（11.2）

求解器：`phase7/placement.py`。输入：BF16 全量 trace 的
`expert_token_histogram.json`（每层每专家 token 数）、实测通信代价矩阵、
专家大小与 NUMA 拓扑。输出 `expert_to_gpu`、副本、预测跨 NUMA bytes、
预测 p99、trace hash。

四卡（NUMA0=GPU0-1，NUMA1=GPU2-3）策略对比：

| 策略 | 负载不均衡 | 预测 p99 | 迁移字节 |
|---|---:|---:|---:|
| static_linear | 3.05% | 0.234 ms | 0 |
| static_round_robin | 2.26% | 0.232 ms | 0.40 GB |
| load_only | 0.02% | 0.227 ms | 0.40 GB |
| load_topology | 0.02% | 0.227 ms | 0.42 GB |

load-aware 策略把不均衡从 2-3% 压到 0.02%，预测 p99 略降（0.234 →
0.227 ms），代价是相对 static 的 ~0.4 GB 迁移（256 专家 × 4MB × ~40% 移动）。

## 3. 结论与边界

1. **负载感知 placement 显著改善均衡**（不均衡 2-3% → 0.02%），且迁移
   成本可量化（单专家 10-79μs，整体 ~0.4GB 字节级迁移）。
2. 跨 NUMA bytes 估计为均匀近似（无 per-token NUMA 归属数据），对四种
   策略数值相同；接入真实 routed ids 后可精确化，属于后续增强点。
3. 11.3 在线控制器（窗口/EMA/触发/收益门限/回滚）已按 Runbook 参数
   定义，待接入真实服务（需要 vLLM EP + 动态路由）后验证。

原始数据：
[placement](Q-TopoMoE_Phase7_placement_20260807.json) /
[migration cost](Q-TopoMoE_Phase7_migration_cost_20260807.json)。
