# Q-TopoMoE 文档索引

这里按“当前结论—实现说明—复现实验—生产边界”组织文档。首次阅读建议先看最终验收，
再按需要进入代码或各阶段报告。

## 最短阅读路径

1. [项目 README](../README.md)：问题、架构、核心实现与量化结果。
2. [Phase 8 最终验收](results/phase8_warm_placement_final_acceptance_20260825.md)：当前候选、canary 与自动闭环结果。
3. [Release manifest](Q-TopoMoE_release_manifest_20260825.json)：推荐模型、候选、运行时补丁、策略和 SHA-256。
4. [代码导读](Q-TopoMoE_代码导读.md)：请求、评测、selector、placement 与闭环控制的调用关系。
5. [复现阅读指南](Q-TopoMoE_复现阅读指南.md)：环境、输入、证据顺序和最小复现路径。

## 当前结论

| 主题 | 已完成内容 | 权威入口 |
|---|---|---|
| 硬件与通信 | 8 卡拓扑、NUMA、P2P 与 NCCL 成本 | [Phase 0 实测](Q-TopoMoE_gpu111_phase0实测分析.md) |
| 服务与量化 | BF16/FP8 基线、W4A16 Triton、RedHatAI NVFP4 准入 | [Phase 1](results/phase1_service_baseline.md)、[Phase 2](results/phase2_quantization_quality.md) |
| 全量质量与路由 | BF16/FP8/NVFP4 共同分母比较与 route trace | [质量收尾](results/phase3_fullset_quality_closeout_20260812.md)、[路由报告](results/phase3_route_and_official_eval.md) |
| kernel 与多卡策略 | M-bucket、backend、TP/DP/EP、通信和迁移成本 | [Phase 4–7](results/phase4_to_phase7_engineering.md) |
| 暖态 placement | 质量、路由稳定性、有限 canary 和自动回滚闭环 | [Phase 8 最终验收](results/phase8_warm_placement_final_acceptance_20260825.md) |
| 部署边界 | 技术验收已完成，生产部署尚未执行 | [发布与生产部署清单](Q-TopoMoE_项目发布与生产部署清单_20260825.md) |

## 实现与复现

| 文档 | 用途 |
|---|---|
| [代码导读](Q-TopoMoE_代码导读.md) | 从入口脚本定位到客户端、聚合器、选择器和运行时补丁 |
| [复现阅读指南](Q-TopoMoE_复现阅读指南.md) | 确认环境、冻结输入、评测口径和证据优先级 |
| [逐步执行 Runbook](Q-TopoMoE_逐步执行Runbook.md) | 按 Phase 0–8 执行并核对每一级准入条件 |
| [总体技术方案](Q-TopoMoE_量化与SM120算子协同优化执行方案.md) | 理解量化、拓扑、kernel、路由与 placement 的依赖关系 |

## 阶段报告

| 阶段 | 报告 |
|---|---|
| Phase 0 | [gpu-111 拓扑与 P2P 实测](Q-TopoMoE_gpu111_phase0实测分析.md) |
| Phase 1 | [BF16/FP8 服务基线](results/phase1_service_baseline.md) |
| Phase 2 | [量化、checkpoint 与质量检查](results/phase2_quantization_quality.md) |
| Phase 3 | [route trace 与 official-like](results/phase3_route_and_official_eval.md)、[full-set 质量收尾](results/phase3_fullset_quality_closeout_20260812.md) |
| Phase 4–7 | [kernel、融合、通信与 EPLB](results/phase4_to_phase7_engineering.md) |
| Phase 8 输入 | [正式矩阵与校准](results/phase8_benchmark_history.md) |
| Phase 8 最终 | [暖态 placement 最终验收](results/phase8_warm_placement_final_acceptance_20260825.md) |

## 证据使用规则

1. 当前发布组合以 [Release manifest](Q-TopoMoE_release_manifest_20260825.json) 为入口。
2. JSON 和 manifest 记录状态、输入身份与哈希；Markdown 负责解释，不替代机器结果。
3. `/data/...`、`/home/...` 和 `artifacts/...` 是服务器或运行输出路径，不是 Git 文件。
4. 大体积 JSONL、日志、模型和 trace 不入库；来源、冻结参数和关键哈希统一记录在
   [Release manifest](Q-TopoMoE_release_manifest_20260825.json)及各阶段机器摘要中。
5. 生产流量切换、扩量和长期容量验证属于独立部署工作，不得由技术验收结果直接推断。
