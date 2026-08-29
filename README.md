# Q-TopoMoE

Q-TopoMoE 面向 8×RTX 5090 PCIe 服务器，研究量化 MoE 推理中的拓扑感知并行、
kernel/backend 选择、专家放置与在线迁移。项目覆盖 BF16、FP8、W4A16 和
NVFP4，并以可复现的服务 Gate、冻结评测集和机器可读结果为准。

## 当前结论

> 更新时间：2026-08-25（Asia/Shanghai）
> 当前状态：实验与技术验收已完成；生产部署未执行。

三格式共同完成的 24,330 条 full-set 样本上，BF16、FP8、NVFP4 准确率分别为
86.9955%、86.9749%、86.3009%。BF16 与 FP8 总体近似持平，NVFP4 相对 BF16
下降 0.6946 个百分点。

| 阶段 | 已完成结论 | 当前状态 |
|---|---|---|
| Phase 0 | 8 卡拓扑、NUMA、NCCL/P2P 实测 | 已完成；P2P 已生效 |
| Phase 1 | BF16/FP8 服务基线、矩阵与统计 | 已完成 |
| Phase 2 | W4A16/NVFP4 审计、真实加载和 official-like Gate | 已完成；正式路线采用 RedHatAI NVFP4 |
| Phase 3 | route trace、漂移分析和冻结官方协议 | route、三格式 full-set 合并与共同分母对比均已完成 |
| Phase 4–6 | M-bucket、kernel/backend selector、通信矩阵 | 正式实测已归档 |
| Phase 7 | P2P 迁移微基准与 placement 实现 | 已完成；部署计划由 Phase 8 暖态链生成 |
| Phase 8 | 暖态 placement、质量、路由稳定性、有限 canary 与自动闭环 | 技术 Gate 已接受 |

Phase 8 最终采用 `warm_swap_008_slots_per_layer_v1`：移动 320/10240 个槽位；
116 题质量 A/B/A、当前 1% 路由稳定性 Gate、384 请求有限 canary，以及 20 窗口、
2560 请求的 trigger/cooldown/apply/rollback 自动闭环均已接受。该结论只表示技术验证完成，
扩大流量、长期运行和生产切换仍是独立变更。

被拒绝但合同有效的候选和被替代政策不属于当前结论；处理理由统一记录在
[实验决策记录](docs/DECISIONS.md)。

正式技术验收版本冻结为 `qtopomoe-phase8-accepted-20260825`；部署前置条件、扩量顺序和
回滚标准见[项目发布与生产部署清单](docs/Q-TopoMoE_项目发布与生产部署清单_20260825.md)。

## 从哪里开始

| 目的 | 推荐入口 |
|---|---|
| 快速了解当前状态 | [文档索引](docs/README.md)与[Phase 8 最终验收](docs/results/phase8_warm_placement_final_acceptance_20260825.md) |
| 独立接管和操作项目 | [项目接管与操作手册](docs/Q-TopoMoE_项目接管与操作手册_20260813.md) |
| 了解路线取舍和失败分支 | [实验决策记录](docs/DECISIONS.md) |
| 回顾完整执行顺序与故障处置 | [项目执行全史与问题处置](docs/Q-TopoMoE_项目执行全史与问题处置_20260813.md) |
| 从头复现 | [复现阅读指南](docs/Q-TopoMoE_复现阅读指南.md) |
| 看懂代码调用关系 | [代码导读](docs/Q-TopoMoE_代码导读.md) |
| 按阶段执行 | [逐步执行 Runbook](docs/Q-TopoMoE_逐步执行Runbook.md) |
| 查某个 JSON/配置的含义与哈希 | [数据与结果清单](docs/DATA_CATALOG.md) |
| 查看三格式全量质量结论 | [BF16/FP8/NVFP4 full-set 收尾](docs/results/phase3_fullset_quality_closeout_20260812.md) |
| 查看 Phase 8 最终结论 | [暖态 placement 最终验收报告](docs/results/phase8_warm_placement_final_acceptance_20260825.md) |
| 查看最终发布与生产部署边界 | [项目发布与生产部署清单](docs/Q-TopoMoE_项目发布与生产部署清单_20260825.md) |

## 快速检查

服务器上的当前仓库根目录是 `/data/moe`：

```bash
cd /data/moe
source env/activate.sh
make help
make check
qtopomoe_gpu_status
```

选择隔离的服务环境：

```bash
qtopomoe_use_vllm
# 或在新 shell 中
qtopomoe_use_sglang
```

`env/activate.sh` 只设置项目、CUDA 和 NCCL 环境，不会占用 GPU 或启动服务。

## 目录职责

| 目录 | 职责 | 主要入口或产物 |
|---|---|---|
| `configs/` | 冻结的模型、拓扑、workload、策略与评测配置 | `configs/experiments/README.md`、`configs/evaluation/README.md` |
| `env/` | 项目变量、环境检查与依赖锁 | `env/activate.sh`、`env/check_env.sh` |
| `topology/` | GPU/NUMA/P2P/NCCL 采集与成本模型 | `topology/collect_hardware.sh`、`topology/gpu_peer_bf16.py` |
| `quantization/` | W4A16/NVFP4 量化及 checkpoint 审计 | `quantization/audit_nvfp4.py` |
| `serving/` | 独立服务启动与四级服务验收 | `serving/start_server.sh`、`serving/acceptance.sh` |
| `clients/` | smoke、质量和工作负载客户端 | `clients/quality_eval.py` |
| `traces/`、`analysis/` | 路由采集与漂移分析 | `traces/capture_routes.py`、`analysis/route_drift.py` |
| `phase4/`、`selector/` | kernel 数据结构和 backend/策略选择逻辑 | `selector/backend_selector.py`、`selector/strategy_selector.py` |
| `phase7/` | 专家放置与迁移成本模型 | `phase7/placement.py`、`phase7/migration_cost.py` |
| `scripts/` | 跨阶段执行、聚合、审计与正式 runner | 见下方常用命令 |
| `evaluation/` | 评测资产抓取、冻结、切片和质量比较 | `evaluation/freeze_full_set_official.py` |
| `tests/` | 不依赖大模型权重的单元/结构测试 | `python -m unittest discover -s tests` |
| `docs/` | 当前结论、决策记录、历史审计与机器结果 | `docs/README.md`、`docs/DECISIONS.md` |

## 常用执行入口

```bash
# 配置和结构检查
python scripts/validate_configs.py
python scripts/check_document_references.py

# checkpoint 静态/真实加载 Gate（示例；输出目录必须是新目录）
BACKEND=vllm MODEL_PATH=<checkpoint> OUT_DIR=<新输出目录> \
  bash scripts/gate_qwen35_checkpoint.sh

# 可断点续跑的 full-set A/B 双分片
nohup setsid bash scripts/run_fullset_quality_pair.sh nvfp4 <输出目录> \
  > <输出目录>.manager.log 2>&1 < /dev/null &

# 严格合并基础轮与截断续跑；重复、缺失、越权替换或身份不一致都会直接失败
python scripts/merge_fullset_quality_results.py --help

# 在同一可评分 ID 上比较两个或多个格式
python scripts/compare_fullset_quality_results.py --help

# Phase 8 当前暖态 placement 验收链
python scripts/build_warm_placement_candidates.py --help
python scripts/run_phase8_warm_placement_abab.py --help
python scripts/run_phase8_warm_placement_quality_equivalence.py --help
python scripts/run_phase8_route_stability_diagnostic.py --help
python scripts/run_phase8_selector_limited_canary.py --help
python scripts/run_phase8_selector_closed_loop_acceptance.py --help

# 在线 placement-plan 与迁移结果分析
python scripts/build_runtime_placement_plan.py --help
python scripts/analyze_online_eplb_gate.py --help
```

正式运行前必须检查 GPU PID、监听端口和输出目录；禁止使用 `pkill python`、
`killall` 或任何无法限定到本项目 PID 的清理命令。

## 证据与复现规则

1. 当前阶段报告说明“结论”，机器可读 JSON/manifest 提供“证据”。二者冲突时，
   以较新的正式 Gate 和其输入哈希为准。
2. `docs/archive/` 只保存被拒绝、被替代或诊断性的历史结果，不得作为当前 Gate。
3. 大体积 JSONL、日志、模型和 trace 不进入 Git；由 manifest 记录路径、版本、
   样本参数和 SHA-256。
4. 机器可读文件保持稳定路径。整理仓库时优先改索引和说明，不随意移动这些文件。
5. 服务器运行目录、模型目录和本地 Git 镜像是三类不同路径，不得混写。

完整数据说明见 [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md)。
