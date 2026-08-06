# Q-TopoMoE 文档索引

所有阶段报告、实测结果与 gate 记录统一归档在本目录。命名约定：
`Q-TopoMoE_<Phase>_<主题>_<YYYYMMDD>.<md|json>`；`.json` 为机器可读结果，
同名 `.md` 为人类可读报告。

## 项目总纲

- [上手与复现阅读指南](Q-TopoMoE_复现阅读指南.md)：新接手者的文件阅读顺序与各文件作用。
- [逐步执行 Runbook](Q-TopoMoE_逐步执行Runbook.md)：端到端执行手册（gate、trace、评测、kernel、策略）。
- [量化与 SM120 算子协同优化执行方案](Q-TopoMoE_量化与SM120算子协同优化执行方案.md)：总体方案。

## Phase 0：实机拓扑

- [8×5090 实机拓扑评估与首轮实验矩阵](Q-TopoMoE_8x5090实机拓扑评估与首轮实验矩阵.md)
- [gpu111 phase0 实测分析](Q-TopoMoE_gpu111_phase0实测分析.md)
- [q_topomoe_phase0_verify.sh](q_topomoe_phase0_verify.sh)

## Phase 1：BF16 / FP8 服务实测

- [Phase1 BF16 实测报告](Q-TopoMoE_Phase1_BF16_实测报告.md)
- [Phase1 BF16/FP8 实测报告](Q-TopoMoE_Phase1_BF16_FP8_实测报告.md)
- [Phase1 统计（报告）](Q-TopoMoE_Phase1_statistics_20260804.md) /
  [Phase1 统计（数据）](Q-TopoMoE_Phase1_statistics_20260804.json)

## Phase 2：量化、canonical checkpoint 与质量 gate

### checkpoint / 兼容性

- [BF16 checkpoint manifest](Q-TopoMoE_BF16_checkpoint_manifest.md)
- [Qwen3.5 canonical checkpoint（数据）](Q-TopoMoE_Qwen35_canonical_checkpoint_20260805.json)
- [Qwen3.5 cleanroom pins（数据）](Q-TopoMoE_Qwen35_cleanroom_pins_20260805.json)
- [Qwen3.5 兼容基线](Q-TopoMoE_Qwen35_compat_baseline_20260805.md)
- [canonical tp1 gate（报告）](Q-TopoMoE_Qwen35_canonical_tp1_gate_20260805.md) /
  [canonical tp1 gate（数据）](Q-TopoMoE_Qwen35_canonical_tp1_gate_20260805.json)
- [canonical tp2 smoke（报告）](Q-TopoMoE_Qwen35_canonical_tp2_smoke_20260805.md) /
  [canonical tp2 smoke（数据）](Q-TopoMoE_Qwen35_canonical_tp2_smoke_20260805.json)
- [original upstream gate（报告）](Q-TopoMoE_Qwen35_original_upstream_gate_20260805.md) /
  [original upstream gate（数据）](Q-TopoMoE_Qwen35_original_upstream_gate_20260805.json)

### 量化与 W4A16

- [Phase2 量化预检报告](Q-TopoMoE_Phase2_量化预检报告.md)
- [Phase2 W4A16 freeze（数据）](Q-TopoMoE_W4A16_freeze_20260805.json)
- [Phase2 W4A16 审计（数据）](Q-TopoMoE_Phase2_W4A16_audit_20260805.json)
- [Phase2 W4A16 failure log](Q-TopoMoE_Phase2_W4A16_failure_log.md)
- [Phase2 W4A16 load gate（报告）](Q-TopoMoE_Phase2_W4A16_load_gate_20260805.md) /
  [Phase2 W4A16 load gate（数据）](Q-TopoMoE_Phase2_W4A16_load_gate_20260805.json)
- [Phase2 W4A16 run status](Q-TopoMoE_Phase2_W4A16_run_status.md)

### 质量 smoke 与 backend gate

- [Phase2 quality smoke 与 backend gate](Q-TopoMoE_Phase2_quality_smoke_and_backend_gate_20260805.md)
- quality smoke 数据：BF16 [vllm tp4](Q-TopoMoE_quality_smoke_bf16_vllm_tp4_20260805.json)；
  W4 [sglang tp4](Q-TopoMoE_quality_smoke_w4_sglang_tp4_20260805.json)、
  [vllm marlin tp4](Q-TopoMoE_quality_smoke_w4_vllm_marlin_tp4_20260805.json)、
  [vllm triton tp4](Q-TopoMoE_quality_smoke_w4_vllm_triton_tp4_20260805.json)；
  对比 [marlin](Q-TopoMoE_quality_smoke_compare_marlin_20260805.json)、
  [triton](Q-TopoMoE_quality_smoke_compare_triton_20260805.json)

### official-like v2（116 条冻结样本）

- [BF16 vs W4A16（triton）结果报告](Q-TopoMoE_Phase2_official_like_v2_BF16_W4_results_20260805.md)
- [BF16 summary（数据）](Q-TopoMoE_Phase2_official_like_v2_BF16_summary_20260805.json)
- [W4A16 summary（数据）](Q-TopoMoE_Phase2_official_like_v2_W4A16_summary_20260805.json)
- [BF16 vs W4 compare（数据）](Q-TopoMoE_Phase2_official_like_v2_BF16_vs_W4_compare_20260805.json)

### WikiText 校准

- [WikiText calibration 报告](Q-TopoMoE_Phase2_WikiText_calibration_report.md) /
  [WikiText calibration manifest](Q-TopoMoE_Phase2_WikiText_calibration_manifest.json)

## Phase 3：route trace 采集与漂移分析

- [Phase3 全量 trace 采集与漂移分析报告](Q-TopoMoE_Phase3_route_trace_full_drift_20260806.md)
- [Phase3 全量漂移数据（BF16 vs W4A16-triton）](Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json)
- 采集 manifest：`traces/manifests/bf16_full_capture_manifest.json`、
  `traces/manifests/w4a16_full_capture_manifest.json`
- Full-set 评测资产：`configs/evaluation/full_set_official_v1.manifest.json`

## Phase 3c：full-set 官方协议冻结与计时 pilot

- [Phase3c full-set 官方协议冻结与计时 pilot 报告](Q-TopoMoE_Phase3c_full_set_official_protocol_20260806.md)
- [Phase3c W4A16 full-set 官方协议评测结果](Q-TopoMoE_Phase3c_full_official_W4_results_20260806.md) /
  [合并 summary（数据）](Q-TopoMoE_Phase3c_full_official_w4_merged_summary_20260806.json)
- 官方协议 manifest：`configs/evaluation/full_set_official_protocol_v1.manifest.json`
- 官方协议 pilot 计时（数据）：
  [bf16_tp4 分层摘要](Q-TopoMoE_Phase3c_pilot_official_protocol_bf16_tp4_20260806.json) /
  [quality summary](Q-TopoMoE_Phase3c_pilot_official_protocol_summary_20260806.json)
- thinking 模式对比 pilot（数据）：
  [bf16_tp4 摘要](Q-TopoMoE_Phase3c_pilot_thinking_bf16_tp4_20260806.json) /
  [quality summary](Q-TopoMoE_Phase3c_pilot_thinking_summary_20260806.json)

## Phase 4 / Phase 8：kernel benchmark 与策略回放

- [Phase4/8 可执行框架](Q-TopoMoE_Phase4_Phase8_framework.md)
- [Phase4/8 离线任务](Q-TopoMoE_Phase4_Phase8_offline_tasks.md)
- [Phase8 replay（数据）](Q-TopoMoE_Phase8_replay_20260804.json)
