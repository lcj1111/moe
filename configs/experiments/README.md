# 实验配置索引

配置文件保持原路径，以维持历史 manifest、脚本和 SHA-256 关系。是否属于当前主线不能只看
文件名，应以状态、policy 和 [`../../docs/DECISIONS.md`](../../docs/DECISIONS.md) 为准。

## 当前 Phase 8 验收链

- `phase8_warm_placement_capture_v1.json`：暖态负载采集合同。
- `phase8_warm_placement_abab_v1.json`：同进程暖态 A/B/A/B。
- `phase8_warm_placement_nvfp4_aux_regression_v2.json`：辅助尺度迁移定向回归。
- `phase8_warm_placement_quality_equivalence_v3.json`：最终 116 题质量 A/B/A。
- `phase8_route_stability_diagnostic_v1.json`、`v2.json`：当前 1% policy 的诊断证据。
- `phase8_warm_placement_limited_canary_v1.json`：最终有限 canary。
- `phase8_warm_placement_closed_loop_acceptance_v1.json`：最终自动闭环验收。

## 历史配置

以下前缀对应的配置只用于复现旧分支，不定义当前 Gate：

- `phase8_cache_arrival_pilot_*`
- `phase8_formal_controlled_*`
- `phase8_selector_independent_*`
- `phase8_selector_limited_canary_*`
- `phase8_selector_one_shot_*`
- `phase8_selector_closed_loop_acceptance_*`

历史配置不会删除或改写。当前候选、最终报告和部署边界分别以
`configs/strategies/phase8_warm_swap_008_slots_per_layer_v1.json`、
`docs/results/phase8_warm_placement_final_acceptance_20260825.md` 和
`docs/Q-TopoMoE_项目发布与生产部署清单_20260825.md` 为准。
