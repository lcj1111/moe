# Q-TopoMoE 文档索引

这里区分三类内容：当前结论、复现说明和历史证据。第一次阅读只需要看“当前状态”
和“执行入口”；按文件查数据时再使用数据清单。

## 当前状态

| 主题 | 状态 | 文档 |
|---|---|---|
| FP8/NVFP4 full-set | NVFP4 EP4 A/B 双分片运行中；FP8 排队 | [质量收尾](results/phase3_fullset_quality_closeout_20260812.md) |
| Phase 7 在线迁移 | placement-plan、阻塞、恢复和 p99 Gate 已接受 | [阶段 7–8 正式收尾](results/phase7_phase8_formal_closeout_20260812.md) |
| Phase 8 正式矩阵 | 四候选、108 cells、五重复聚合已接受 | [阶段 8 历史与结果](results/phase8_benchmark_history.md) |
| Phase 8 selector | median regret 通过；p95 regret 43.17%，未准入 | [阶段 7–8 正式收尾](results/phase7_phase8_formal_closeout_20260812.md) |

正在运行的结果在完成 Gate 前仅表示进度，不表示最终准确率。

## 执行入口

1. [项目执行全史与问题处置](Q-TopoMoE_项目执行全史与问题处置_20260813.md)：按时间说明完成顺序、阶段依赖、全部已记录问题及应对方式。
2. [复现阅读指南](Q-TopoMoE_复现阅读指南.md)：依赖、冻结输入和验收顺序。
3. [逐步执行 Runbook](Q-TopoMoE_逐步执行Runbook.md)：Phase 0–8 的操作步骤。
4. [量化与 SM120 协同方案](Q-TopoMoE_量化与SM120算子协同优化执行方案.md)：总体技术路线和约束。
5. [数据与结果清单](DATA_CATALOG.md)：每个跟踪产物的用途、复现约束和 SHA-256。

## 阶段报告

| 阶段 | 聚合报告 |
|---|---|
| Phase 0 | [gpu-111 拓扑与 P2P 合并实测](Q-TopoMoE_gpu111_phase0实测分析.md) |
| Phase 1 | [BF16/FP8 服务基线](results/phase1_service_baseline.md) |
| Phase 2 | [量化、canonical checkpoint 与质量 Gate](results/phase2_quantization_quality.md) |
| Phase 3 | [route trace 与 official-like](results/phase3_route_and_official_eval.md)、[full-set 收尾](results/phase3_fullset_quality_closeout_20260812.md) |
| Phase 4–7 | [kernel、融合、通信与 EPLB](results/phase4_to_phase7_engineering.md) |
| Phase 7–8 | [在线迁移与四候选正式收尾](results/phase7_phase8_formal_closeout_20260812.md) |
| Phase 8 | [基准测试与校准历史](results/phase8_benchmark_history.md) |

在线迁移的独立实验说明见[在线 placement-plan 与服务迁移实验](Q-TopoMoE_在线placement-plan与服务迁移实验_20260812.md)。

## 交接与历史资料

- [NVFP4 2026-08-09 交接快照](HANDOFF_20260809_NVFP4.md)：保留当时环境和故障背景；
  文件顶部列出 2026-08-12 之后的权威状态，旧路径和旧提交号仅用于历史审计。
- `archive/`：失败、被替代或仅供对照的机器可读结果。
- 根目录下日期化 JSON：仍被脚本、报告或数据清单引用的稳定证据，不能只因文件多而移动。

### 文件名阅读规则

- `Q-TopoMoE_Phase<N>_<主题>_<日期>.json`：某次已冻结的机器结果或 Gate；日期越新不一定越权威，还要检查 `status` 和对应报告。
- `*_launch_*.json`：启动计划或运行登记，不等于实验已通过。
- `*_rejected_*.json`：保留的失败证据，不得作为候选准入结果。
- `*_manifest.json`：来源、revision、样本或文件哈希；复现前应先验证。
- `results/phase*.md`：面向阅读的阶段聚合结论，优先于逐个打开根目录 JSON。

## 信息优先级

同一事实出现多个版本时，按以下顺序判断：

1. 最新日期的正式 Gate JSON及其输入哈希；
2. `results/` 中对应阶段的最新收尾报告；
3. Runbook 中的预注册门槛和流程；
4. 日期化交接文档；
5. `archive/` 中的历史结果。

评测输入的来源、revision、样本索引、tokenizer/chat template 和 SHA-256 统一从
[数据与结果清单](DATA_CATALOG.md)及 `configs/evaluation/` manifest 查找。
