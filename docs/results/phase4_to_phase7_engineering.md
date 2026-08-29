# 阶段 4–7：kernel、融合、通信与 EPLB 工程

本文合并 Phase 4–7 的 kernel、融合、通信、EPLB 和早期 selector 工程记录。旧拆分稿
已经删除；可复核数据集中在 `docs/` 下对应阶段 JSON，执行入口集中在 `scripts/`、
`phase4/`、`phase7/` 和 `selector/`。

## Phase 4：kernel 与策略回放

# Q-TopoMoE Phase 4/8 进展：Triton MoE kernel 实测与策略回放

> 生成日期：2026-08-06（Asia/Shanghai）
> 摘要：Phase 4 kernel DB 从空模板首次获得 8 条真实 Triton MoE kernel
> 实测（vLLM `fused_experts`，服务同款）；Phase 8 回放从
> `blocked_missing_kernel_measurements` 转为 `completed`，并由真实 Phase 3
> trace 生成 workload observation 驱动。

## 1. Phase 4：Triton MoE kernel 实测

基准脚本：`scripts/bench_moe_kernel.py`；原始数据：
[Q-TopoMoE Phase4 Triton BF16 kernel v3](../Q-TopoMoE_Phase4_triton_moe_bf16_kernel_20260806_v3.json)。

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
[Phase 8 P2P replay data](../Q-TopoMoE_Phase8_replay_20260807_p2p.json)。

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

---

## Phase 5：permute 与量化融合

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
[Q-TopoMoE_Phase5_permute_quant_fusion_20260806.json](../Q-TopoMoE_Phase5_permute_quant_fusion_20260806.json)。

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

A/B 数据：[Q-TopoMoE_Phase5_prepare_ab_20260807.json](../Q-TopoMoE_Phase5_prepare_ab_20260807.json)。

---

## Phase 6：TP/DP/EP 通信矩阵

# Q-TopoMoE Phase 6：TP/DP/EP 系统矩阵（P2P 后全量）

> 生成日期：2026-08-07（Asia/Shanghai）
> 范围：4×RTX 5090（NUMA0: GPU0-3）与 8×RTX 5090（双 NUMA）。
> 所有数字均来自 P2P 正式环境。

> **2026-08-09 并行拓扑审计更正：** vLLM 的 EP rank 数由实际并行
> world size 决定，不由 `CUDA_VISIBLE_DEVICES` 数量决定。历史
> `ep8_tp1_p2p` 实际 `world_size=1`，不得作为 EP8 结果；历史
> `ep4_tp2_eplb_*` 实际 `world_size=2`，是 EP2+EPLB，不是 EP4。
> 下表保留其时序用于失败审计，但已修正准入含义。

## 1. 环境与前置

- vLLM cleanroom `33c50587d`，支持 `--tensor-parallel-size`、
  `--data-parallel-size`、`--enable-expert-parallel`、`--all2all-backend`
  （`allgather_reducescatter`）、`--enable-eplb`。
- BF16：Qwen3.6-35B-A3B snapshot；W4A16 canonical（triton MoE）。
- 每格：启动 → health → 32 请求 smoke（并发 8，256 in / 64 out）→ summary。
- 执行脚本：`scripts/run_phase6_matrix.sh`。

## 2. 四卡矩阵（P2P 后）

| 配置 | 模型 | TTFT p50 | TPOT p50 | e2e p50 | 通过 |
|---|---|---:|---:|---:|---:|
| TP4×DP1 | BF16 | 190.0 ms | 5.80 ms | 620.5 ms | 32/32 |
| TP2×DP2 | BF16 | 214.0 ms | 6.40 ms | 693.1 ms | 32/32 |
| TP1×DP4 | W4A16 | 238.3 ms | 11.32 ms | 1,010.2 ms | 32/32 |
| EP4 static | W4A16 | 194.0 ms | 8.17 ms | 711.6 ms | 32/32 |

原始数据：[Q-TopoMoE_Phase6_matrix_4gpu_p2p_20260807.json](../Q-TopoMoE_Phase6_matrix_4gpu_p2p_20260807.json)。

四卡 smoke 中，TP4×DP1 的 e2e p50 最低；TP2×DP2 次之。TP1×DP4
证明 W4A16 canonical 可单卡承载，EP4 static 则提供单 NUMA 专家并行基线。

## 3. 八卡矩阵（P2P 后）

| 配置 | 模型 | TTFT p50 | TPOT p50 | e2e p50 | 通过 |
|---|---|---:|---:|---:|---:|
| TP8×DP1 | BF16 | 173.3 ms | 6.25 ms | 627.6 ms | 32/32 |
| TP4×DP2 | BF16 | 316.6 ms | 7.86 ms | 785.4 ms | 32/32 |
| TP2×DP4 | BF16 | 276.3 ms | 8.27 ms | 1,041.6 ms | 32/32 |
| TP1+EP（实际 world=1，非 EP8） | W4A16 | 227.4 ms | 12.54 ms | 1,024.6 ms | 无效拓扑 |
| TP2+EP2+原生 EPLB | W4A16 | 182.2 ms | 16.28 ms | 1,221.5 ms | 32/32 |
| TP2+EP2+原生 EPLB（重复） | W4A16 | 199.1 ms | 15.87 ms | 1,200.7 ms | 32/32 |

原始数据：[Q-TopoMoE_Phase6_matrix_8gpu_p2p_20260807.json](../Q-TopoMoE_Phase6_matrix_8gpu_p2p_20260807.json)。

> 注：EP2+EPLB 行额外做了一次重复运行（run2）以检查稳定性，e2e 1,221.5
> →1,200.7 ms（-1.7%），TTFT/TPOT 波动在 smoke 样本正常范围。

## 4. 观察

1. **TPOT（稳态 decode）**：TP4×DP1 最低（5.80 ms），TP8×DP1 次之
   （6.25 ms）；TP1×DP4 较高（11 ms）。原标注 EP8 的 12.54 ms
   实际是 world=1，不参与 EP 比较。
2. **TTFT（prefill）**：TP8×DP1 最优（173 ms）；TP4×DP2 明显偏高
   （317 ms），DP 协调 + 更细 TP 分片对 prefill 不利。
3. **e2e p50**：TP4×DP1 最低（620 ms）；8 卡矩阵中 TP8×DP1（628 ms）
   与 TP4×DP1 相当，TP2×DP4 与已验证 EP2/EP4 配置略高。
4. TP1×DP4 证明 W4A16 canonical 单卡容量通过（Runbook 前置条件），
   为 TP1×DP8 提供依据。

## 5. 边界与下一步

- 本矩阵为 32 请求 smoke 级正确性 + 时序；正式性能需更大并发与多轮。
- TP1×DP8（W4A16）仍受静态 group scale 对齐限制（group_size=128 与
  DP8 每分区 64 不整除），可用 block64 变体（`..._g64`）支持；TP8×FP8
  同理受 block 对齐限制，保留为格式不兼容证据。
- EP 准入顺序：W4A16 的 EP4/单 NUMA static 已完成；本历史矩阵没有
  有效 EP8 static。原生 EPLB 仅在 EP2（TP2×DP1）完成 smoke，不能外推为
  EP8+原生 EPLB。EP8+冗余专家 1 仍受框架限制：vLLM 要求专家数可被 EP rank 整除，
  256+1=257 为质数，EP2/4/8 均不可用，`--eplb-config
  '{"num_redundant_experts": 1}'` 报 `even distribution of experts across
  ranks`，保留为格式不兼容证据）、自研 topology-aware EPLB（待做）。
---

## Phase 7：专家迁移与 placement

# Q-TopoMoE Phase 7：量化感知 EPLB（离线 placement 与迁移成本）

> 生成日期：2026-08-07（Asia/Shanghai）
> 范围：11.1 迁移成本实测 + 11.2 离线 placement 求解器与策略对比；
> 11.3 在线控制器参数框架在后续阶段接入真实服务后验证。

## 1. 迁移成本实测（11.1）

单个专家 w1+w2（BF16，2048×512×2 字节 = 4MB）的 D2D 拷贝中位耗时
（RTX 5090）：

| 路径 | 耗时/专家 |
|---|---:|
| 同 GPU（intra-device） | 10.14 μs |
| 同 NUMA（peer，GPU0→1） | 47.71 μs |
| 跨 NUMA（peer，GPU0→4） | 73.45 μs |

跨 NUMA 迁移明显慢于同 NUMA peer 拷贝。计划生成必须同时约束迁移数量和
NUMA 边界，不能把两类路径视为同一成本。
测量脚本：`phase7/migration_cost.py`。

## 2. 离线 placement（11.2）

求解器 `phase7/placement.py` 保留为通用实现。输入包括逐层专家 token
直方图、当前 NCCL 成本库、专家大小、NUMA 拓扑和显存预算；输出
`expert_to_gpu`、预测迁移字节与 trace hash。仓库不再提供可直接部署的早期
Phase 7 map，当前唯一正式候选由 Phase 8 真实暖态计数重新生成。

## 3. 结论与边界

1. 迁移成本使用 [P2P 正式微基准](../Q-TopoMoE_Phase7_migration_cost_p2p_20260807.json)。
2. 离线预测不能替代在线稳定性、质量和恢复测试。
3. 当前部署候选与完整 apply/rollback 结论见
   [暖态 placement 最终验收](phase8_warm_placement_final_acceptance_20260825.md)。

---

## 阶段 4 / 8 控制平面与离线复现入口

该框架实现 CPU 侧控制平面，不把计划值伪装成 CUDA 实测。kernel 数据行只有
同时满足 `measured: true`、`valid: true` 且包含 p50 或 p95 延迟时才允许被
默认选择；缺失或无效数据会抛出 `SelectionError`，`allow_unmeasured=True`
只能用于显式开发 dry-run。

### M-bucket 工作负载

```bash
python3 phase4/workload/generate_m_buckets.py \
  --output configs/workloads/m_buckets.json --seed 42
```

manifest 含 108 个确定性 case：W1（256/128，C=1/8/32/128）、W2
（2048/256，C=1/8/32）、W3（8192/256，C=1/8/16）、W4（32768/128，
C=1/4），再与 prefix-cache 0/50/100% 和 closed-loop/Poisson/burst 到达模式
交叉。`prefill_m=input_tokens*concurrency`、`decode_m=concurrency`，向上取整到
二次幂 bucket；`real_M_hist` 保存 prefill/decode 混合分布。

Phase 4 计划生成入口：

```bash
python3 scripts/plan_phase4_kernel_benchmark.py \
  --m-buckets configs/workloads/m_buckets.json \
  --output configs/kernels/phase4_benchmark_plan.json
```

原始计划为 8 个 M bucket × 3 backend × 2 precision，共 48 个 `planned` 单元；
它只生成计划，不执行 CUDA。实测后应把 backend、kernel config、M bucket、
precision、p50/p95、`measured=true` 与 source run ID 写入
`configs/kernels/phase4_kernel_db.json`。

官方 RedHatAI NVFP4 的 `vllm.run_cutlass_moe_fp4` 已有 50 次重复实测：
M=1/2048/8192 的 p95 分别为 250.42/728.23/2219.58 μs；补充 bucket
M=4/8/16/32/128/256/16384 的 p50/p95 分别为 235.61/253.71、
232.72/240.26、285.39/293.39、361.28/369.05、453.48/461.00、
475.03/485.84、4382.33/4413.00 μs。机器数据见
`docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_real_m_20260809.json` 与
`docs/Q-TopoMoE_Phase4_NVFP4_CUTLASS_phase8_missing_m_20260809.json`。
这些直接 kernel 测量不能替代 EP 服务数据，因为冻结 vLLM 在专家分片服务中
选择的是 MARLIN。

### 通信成本和策略回放

```bash
python3 scripts/build_nccl_cost_db.py \
  --input artifacts/raw/20260807T120000Z_nccl_formal_p2p/nccl/statistics.json \
  --output configs/communication/nccl_cost_db.json

python3 scripts/replay_phase8.py \
  --candidates configs/strategies/phase8_candidates.json \
  --kernel-db configs/kernels/phase4_kernel_db.json \
  --observations configs/strategies/phase8_observations.json \
  --output docs/Q-TopoMoE_Phase8_replay_20260807_p2p.json
```

NCCL DB 从 P2P 正式矩阵归一化 `time_us` 实测点，覆盖 6 个 mapping 且
`wrong_total=0`；不同 collective/size 应按点查询或拟合，不能直接压成一个线性
带宽常数。

`StrategyCandidate` 包含量化格式/checkpoint、TP/DP/EP、GPU 映射、EPLB、
冗余 expert、kernel backend/config。`CostModel` 综合真实 M 分布计算、实测通信、
route 不均衡与迁移成本；`evaluate()` 输出 top-1、median/p95 regret、控制器开销占
预测 p99 的比例和无效配置率。Gate 为 median regret ≤5%、p95 ≤10%、
controller overhead <1%；透明模型被证明确实不足前不引入 RL。正式结果、容量
冻结与校准历史见[阶段 8 基准与校准历史](phase8_benchmark_history.md)。
