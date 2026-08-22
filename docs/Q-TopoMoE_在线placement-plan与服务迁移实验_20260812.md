# Q-TopoMoE 在线 placement-plan 应用与服务迁移实验

## 结论

本轮 Gate 已通过。实测 placement plan 已在 vLLM 的 8 个 EP rank 上以同一 SHA-256 应用，并由 vLLM 原生同步 EPLB 完成权重重排、通信屏障和路由映射提交。稳定、迁移、恢复三个阶段共 384 个请求，失败数为 0。

需要特别说明：原离线拓扑计划不能原样上线。它没有约束 vLLM 每层每卡固定 32 个物理专家槽，例如第 0 层 GPU0 被分配 43 个专家。实验没有绕过该约束，而是使用同一份实测 expert-token histogram，通过 vLLM 原生 EPLB policy 生成了 `40 层 × 256 槽`、每卡每层严格 32 槽的运行时计划。

## 实验配置

| 项目 | 配置 |
|---|---|
| 模型 | RedHatAI Qwen3.5 MoE NVFP4 |
| 服务框架 | vLLM `0.26.1rc1.dev343+g33c50587d` |
| 并行 | TP8 + EP8 |
| EPLB | 同步模式，`torch_nccl` communicator |
| 迁移计划 | 40 层、256 专家、8 rank、每 rank 每层 32 槽 |
| 请求负载 | 输入 256 tokens、输出 32 tokens、并发 8、每阶段 128 请求 |
| 随机种子 | 42 |
| EPLB 周期 | window 16、step interval 16 |

## 在线应用证据

- profile 阶段在 8 个 rank 上都返回原始线性布局，没有提前应用目标计划。
- 第一个在线 EPLB 周期在 8 个 rank 上应用同一映射 SHA-256：`d7d7240007c8d69cce9350ff6e4deca250713282c651d32c954b077b383763fd`。
- vLLM 原生日志记录首次在线权重重排耗时 `0.83 s`。
- 计划相对线性初始布局移动 10,187 个物理槽；按每专家 1,769,496 bytes 估算，迁移权重总量为 18,025,855,752 bytes。
- 首次迁移完成后，相同计划的同步检查/提交 p50、p95、p99 均为 `0.08 s`。

## p99 与恢复结果

| 阶段 | 成功/总数 | E2E p50 | E2E p95 | E2E p99 | 均值 |
|---|---:|---:|---:|---:|---:|
| 稳定计划基线 | 128/128 | 874.90 ms | 1072.02 ms | 1082.04 ms | 825.86 ms |
| 在线迁移窗口 | 128/128 | 974.52 ms | 1835.55 ms | 2166.96 ms | 980.76 ms |
| 迁移后恢复 | 128/128 | 797.56 ms | 986.11 ms | 988.83 ms | 805.32 ms |

迁移窗口 p99 相对稳定基线上升 `100.27%`，说明同步迁移阻塞确实进入了用户请求关键路径。恢复窗口 p99 为稳定基线的 `91.39%`，低于“恢复后不超过基线 5%”的 Gate 阈值，因此恢复 Gate 通过。

## Gate 状态

| Gate | 状态 |
|---|---|
| 运行时计划形状与固定槽约束 | 通过 |
| 8 个 rank 使用同一计划 SHA-256 | 通过 |
| 请求期原生权重重排可观测 | 通过 |
| 三阶段请求均无失败 | 通过 |
| 恢复 p99 不超过稳定基线 5% | 通过 |

## 可复现文件

- [完整 Gate 报告](results/Q-TopoMoE_在线EPLB迁移Gate_20260812.json)
- [稳定基线摘要](results/Q-TopoMoE_在线EPLB稳定基线_20260812.json)
- [迁移窗口摘要](results/Q-TopoMoE_在线EPLB迁移窗口_20260812.json)
- [恢复窗口摘要](results/Q-TopoMoE_在线EPLB恢复窗口_20260812.json)
- [运行时 placement plan](../configs/strategies/nvfp4_runtime_eplb_plan_gpu111.json)
- [运行时计划生成脚本](../scripts/build_runtime_placement_plan.py)
- [在线 Gate 分析脚本](../scripts/analyze_online_eplb_gate.py)
- [NVFP4 EPLB 运行时桥接补丁](../runtime_patches/qtopomoe_eplb/sitecustomize.py)
- [服务启动脚本](../scripts/launch_online_eplb_probe.sh)

服务器上的原始请求 JSONL、完整服务日志和进程记录保存在 `/data/models/test/qtopomoe_online_eplb_gate_v1` 与 `/data/models/test/qtopomoe_online_eplb_migration_v1`。这些大体积运行数据不复制进 Git 仓库，摘要文件包含其路径和 SHA-256。
