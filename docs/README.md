# Q-TopoMoE 文档索引

文档按“当前结论—复现入口—历史审计”三层组织。第一次阅读只需要看当前结论；
失败分支和旧版本不会再混入主线报告。

## 当前结论

| 主题 | 当前状态 | 权威入口 |
|---|---|---|
| 8 卡硬件与通信 | 拓扑、NUMA、P2P、NCCL 成本已冻结 | [Phase 0 实测](Q-TopoMoE_gpu111_phase0实测分析.md) |
| 服务与量化 | BF16/FP8 基线、W4A16 Triton、官方 RedHatAI NVFP4 已完成 Gate | [Phase 1](results/phase1_service_baseline.md)、[Phase 2](results/phase2_quantization_quality.md) |
| 全量质量与路由 | BF16/FP8/NVFP4 full-set 共同分母和 route trace 已完成 | [质量收尾](results/phase3_fullset_quality_closeout_20260812.md) |
| kernel、通信与迁移 | M-bucket、backend、TP/DP/EP、迁移成本和在线 bridge 已完成 | [Phase 4–7](results/phase4_to_phase7_engineering.md) |
| Phase 8 | 暖态 placement 的质量、当前 1% 路由 Gate、有限 canary 和自动闭环已接受 | [最终验收](results/phase8_warm_placement_final_acceptance_20260825.md) |
| 部署边界 | 技术验证已完成，生产部署未执行 | [发布与部署清单](Q-TopoMoE_项目发布与生产部署清单_20260825.md) |

## 复现与接管

1. [复现阅读指南](Q-TopoMoE_复现阅读指南.md)：先确认环境、冻结输入和证据优先级。
2. [逐步执行 Runbook](Q-TopoMoE_逐步执行Runbook.md)：按 Phase 0–8 执行。
3. [项目接管与操作手册](Q-TopoMoE_项目接管与操作手册_20260813.md)：服务、GPU、恢复与故障定位。
4. [代码导读](Q-TopoMoE_代码导读.md)：请求、评测、selector 和闭环控制调用关系。
5. [数据与结果清单](DATA_CATALOG.md)：机器产物、冻结配置和 SHA-256。

## 当前阶段报告

| 阶段 | 聚合报告 |
|---|---|
| Phase 0 | [gpu-111 拓扑与 P2P 实测](Q-TopoMoE_gpu111_phase0实测分析.md) |
| Phase 1 | [BF16/FP8 服务基线](results/phase1_service_baseline.md) |
| Phase 2 | [量化、canonical checkpoint 与质量 Gate](results/phase2_quantization_quality.md) |
| Phase 3 | [route trace 与 official-like](results/phase3_route_and_official_eval.md)、[full-set 收尾](results/phase3_fullset_quality_closeout_20260812.md) |
| Phase 4–7 | [kernel、融合、通信与 EPLB](results/phase4_to_phase7_engineering.md) |
| Phase 8 基准输入 | [正式矩阵与校准历史](results/phase8_benchmark_history.md) |
| Phase 8 候选形成 | [暖态 placement 生成与质量验证](results/phase8_warm_placement_regeneration_20260823.md) |
| Phase 8 最终结论 | [暖态 placement 最终验收](results/phase8_warm_placement_final_acceptance_20260825.md) |

## 失败分支与历史证据

[实验决策记录](DECISIONS.md)用一张表解释每个失败或被替代分支为何关闭、由什么结果接替。
需要查看完整过程时，再进入 [`archive/decision-history/`](archive/decision-history/README.md)。

- `rejected` 表示实验有效但候选未通过 Gate。
- `superseded` 表示结论被更严格或环境一致的实验覆盖。
- `diagnostic only` 只用于定位问题，不支持性能收益。
- `invalid run` 不满足实验合同，不并入正式数据。

机器 JSON、manifest 和冻结配置仍保持稳定路径，由 [DATA_CATALOG.md](DATA_CATALOG.md)
索引；文档整理不会删除失败证据或改写原始 Gate。

## 路径与证据优先级

同一事实存在多个版本时，按以下顺序判断：

1. 当前最终 Gate JSON及其输入哈希；
2. `results/` 中当前阶段聚合报告；
3. Runbook 和冻结 policy；
4. [实验决策记录](DECISIONS.md)；
5. `archive/` 中的历史结果。

`/data/...`、`/home/...` 和 `artifacts/...` 是服务器或运行输出路径，不是 Git 文件。
评测来源、revision、样本索引、tokenizer/chat template 和哈希统一以数据清单和
`configs/evaluation/` manifest 为准。
