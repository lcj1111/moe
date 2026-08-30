# 实验配置索引

有效配置保持原路径，以维持 manifest、脚本和 SHA-256 关系。是否属于当前主线不能只看
文件名，应以状态、policy 和
[`../../docs/Q-TopoMoE_release_manifest_20260825.json`](../../docs/Q-TopoMoE_release_manifest_20260825.json) 为准。

## 当前 Phase 8 验收链

- `phase8_warm_placement_capture_v1.json`：暖态负载采集合同。
- `phase8_warm_placement_nvfp4_aux_regression_v2.json`：辅助尺度迁移定向回归。
- `phase8_warm_placement_quality_equivalence_v3.json`：最终 116 题质量 A/B/A。
- `phase8_warm_placement_limited_canary_v1.json`：最终有限 canary。
- `phase8_warm_placement_closed_loop_acceptance_v1.json`：最终自动闭环验收。

其他配置服务于基准构建、校准或独立测量，不单独定义当前发布状态。当前候选、最终报告和部署边界分别以
`configs/strategies/phase8_warm_swap_008_slots_per_layer_v1.json`、
`docs/results/phase8_warm_placement_final_acceptance_20260825.md` 和
`docs/Q-TopoMoE_项目发布与生产部署清单_20260825.md` 为准。
