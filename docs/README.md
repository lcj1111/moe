# Q-TopoMoE 文档索引

本目录按“少量稳定入口 + 集中结果 + 集中数据清单”组织。机器可读的 JSON/JSONL/YAML/manifest 保留原路径，避免影响复现；叙述性报告按阶段合并到 `results/`。

## 先读这三份

- [复现阅读指南](Q-TopoMoE_复现阅读指南.md)：按依赖、冻结输入和验收顺序阅读。
- [逐步执行 Runbook](Q-TopoMoE_逐步执行Runbook.md)：从 Phase 0 到 Phase 8 的执行入口。
- [量化与 SM120 算子协同优化执行方案](Q-TopoMoE_量化与SM120算子协同优化执行方案.md)：总体技术路线与约束。
- [NVFP4 当前交接](HANDOFF_20260809_NVFP4.md)：当前阶段状态、Gate 和下一步。

## 阶段结果（叙述报告）

- [Phase 1：BF16/FP8 服务基线](results/phase1_service_baseline.md)
- [Phase 2：量化、canonical checkpoint 与质量 Gate](results/phase2_quantization_quality.md)
- [Phase 3：route trace 与 official-like 评测](results/phase3_route_and_official_eval.md)
- [Phase 4–7：kernel、融合、通信与 EPLB 工程](results/phase4_to_phase7_engineering.md)
- [阶段 8：基准测试与校准历史](results/phase8_benchmark_history.md)

## 数据与机器可读输入

- [数据与结果清单](DATA_CATALOG.md)：逐文件说明类型、用途、复现约束和 SHA-256。
- [评测 README](../configs/evaluation/README.md)：冻结评测输入、manifest、样本集和 hash 约束。
- [Phase 0 拓扑与 P2P](Q-TopoMoE_8x5090实机拓扑评估与首轮实验矩阵.md)、[gpu111 实测分析](Q-TopoMoE_gpu111_phase0实测分析.md)、[P2P 重测总结](Q-TopoMoE_P2P_retest_summary_20260807.md)。

## 归档规则

`docs/archive/` 仅保存历史或被拒绝的对照数据；当前 Gate 不得直接引用归档记录。所有原始合并文件仍可从 Git 历史恢复，合并报告内记录了源文件的大小与 SHA-256。
