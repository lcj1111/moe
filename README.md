# Q-TopoMoE

Q-TopoMoE 面向 8×RTX 5090 PCIe 服务器，研究量化 MoE 推理中的拓扑感知并行、
kernel/backend 选择、专家放置与在线迁移。项目覆盖 BF16、FP8、W4A16 和
NVFP4，并以可复现的服务 Gate、冻结评测集和机器可读结果为准。

## 当前结论

> 更新时间：2026-08-16（Asia/Shanghai）
> 当前状态：BF16、FP8、NVFP4 full-set 均已完成基础轮、有限续跑、严格合并与
> 三格式共同分母比较。共同完成的 24,330 条上三者分别为 86.9955%、86.9749%、
> 86.3009%；BF16 与 FP8 总体近似持平，NVFP4 相对 BF16 下降 0.6946 个百分点。

| 阶段 | 已完成结论 | 当前状态 |
|---|---|---|
| Phase 0 | 8 卡拓扑、NUMA、NCCL/P2P 实测 | 已完成；P2P 已生效 |
| Phase 1 | BF16/FP8 服务基线、矩阵与统计 | 已完成 |
| Phase 2 | W4A16/NVFP4 审计、真实加载和 official-like Gate | 已完成；RedHatAI NVFP4 准入，自生成 NVFP4 v1 拒绝 |
| Phase 3 | route trace、漂移分析和冻结官方协议 | route、三格式 full-set 合并与共同分母对比均已完成 |
| Phase 4–6 | M-bucket、kernel/backend selector、通信矩阵 | 正式实测已归档 |
| Phase 7 | placement-plan、在线迁移、恢复与 p99 | Gate 已接受 |
| Phase 8 | 四候选×108 cells×5 重复正式聚合 | v3 训练 p95 regret 已由 23.05% 降至 3.49%，训练 Gate 已冻结；独立 900 次 Gate 待完成 |

动态触发、cooldown 和 rollback 闭环必须等待 Phase 8 selector 独立 900 次 Gate
通过；训练 Gate 通过不等于已经具备在线控制资格。

## 从哪里开始

| 目的 | 推荐入口 |
|---|---|
| 快速了解当前状态 | [文档索引](docs/README.md)与[阶段 7–8 正式收尾](docs/results/phase7_phase8_formal_closeout_20260812.md) |
| 独立接管和操作项目 | [项目接管与操作手册](docs/Q-TopoMoE_项目接管与操作手册_20260813.md) |
| 回顾完整执行顺序与故障处置 | [项目执行全史与问题处置](docs/Q-TopoMoE_项目执行全史与问题处置_20260813.md) |
| 从头复现 | [复现阅读指南](docs/Q-TopoMoE_复现阅读指南.md) |
| 按阶段执行 | [逐步执行 Runbook](docs/Q-TopoMoE_逐步执行Runbook.md) |
| 查某个 JSON/配置的含义与哈希 | [数据与结果清单](docs/DATA_CATALOG.md) |
| 查看三格式全量质量结论 | [BF16/FP8/NVFP4 full-set 收尾](docs/results/phase3_fullset_quality_closeout_20260812.md) |
| 查看 Phase 8 selector 最新结论 | [selector v3 训练 Gate 通过报告](docs/results/phase8_selector_v3_training_gate_accepted_20260816.md) |

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
| `configs/` | 冻结的模型、拓扑、workload、策略与评测配置 | `configs/evaluation/README.md` |
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
| `docs/` | 当前结论、历史报告与机器可读结果 | `docs/README.md` |

## 常用执行入口

```bash
# 配置和结构检查
python scripts/validate_configs.py

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

# Phase 8 正式重复测量与聚合
python scripts/run_phase8_repeated.py --help
python scripts/aggregate_phase8_repeated.py --help
python scripts/run_phase8_predecision_windows.py --help
python scripts/fit_phase8_selector_state.py --help
python scripts/evaluate_phase8_independent_selector.py --help

# 在线 placement-plan 与迁移结果分析
python scripts/build_runtime_placement_plan.py --help
python scripts/analyze_online_eplb_gate.py --help
```

正式运行前必须检查 GPU PID、监听端口和输出目录；禁止使用 `pkill python`、
`killall` 或任何无法限定到本项目 PID 的清理命令。

## 证据与复现规则

1. 当前阶段报告说明“结论”，机器可读 JSON/manifest 提供“证据”。二者冲突时，
   以较新的正式 Gate 和其输入哈希为准。
2. `docs/archive/` 只保存被拒绝或被替代的历史结果，不得作为当前 Gate。
3. 大体积 JSONL、日志、模型和 trace 不进入 Git；由 manifest 记录路径、版本、
   样本参数和 SHA-256。
4. 机器可读文件保持稳定路径。整理仓库时优先改索引和说明，不随意移动这些文件。
5. 服务器运行目录、模型目录和本地 Git 镜像是三类不同路径，不得混写。

完整数据说明见 [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md)。
