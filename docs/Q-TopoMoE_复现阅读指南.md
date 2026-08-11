# Q-TopoMoE 上手与复现阅读指南

> 适用分支：`agent/sync-q-topomoe-project`
> 目标：让一位新接手者按本指南顺序读完文件后，能理解项目全貌并独立复现
> BF16/W4A16 质量评测、route trace 采集与漂移分析、full-set 官方协议评测。

## 阅读顺序总览

```
第 1 步  项目定位     README.md → docs/README.md
第 2 步  方案与手册   Runbook → 执行方案 → Phase 4/8 框架
第 3 步  已有成果     docs/ 下 Phase 0/1/2/3/3c 报告（按阶段）
第 4 步  环境与配置   env/ → Makefile → configs/ 说明
第 5 步  复现主线     gate → quality_eval → freeze 系列 → trace → drift
第 6 步  专项深入     quantization/ → topology/ → selector/ → phase4/
第 7 步  测试验证     tests/ → scripts/validate_configs.py
```

---

## 第 1 步：项目定位（2 个文件）

| 文件 | 作用 |
|---|---|
| `README.md` | 项目入口：定位、当前阶段进度表、目录约定、快速开始、复现流程、安全边界。先读这里建立全局印象。 |
| `docs/README.md` | 文档索引：把所有报告按 Phase 0-8 归类，含指向 `.md` 报告与 `.json` 原始数据的链接。找任何历史结果都从这页进。 |

## 第 2 步：方案与手册（3 个文件）

| 文件 | 作用 |
|---|---|
| `docs/Q-TopoMoE_逐步执行Runbook.md` | 端到端执行手册：从环境、gate、trace、评测到 Phase 4/8 的每一步命令与验收标准。复现前必须通读。 |
| `docs/Q-TopoMoE_量化与SM120算子协同优化执行方案.md` | 总体技术方案：量化感知 MoE 并行、动态负载均衡、SM120 算子的设计意图与阶段划分。 |
| `docs/Q-TopoMoE_Phase4_Phase8_framework.md` | Phase 4/8 可执行框架的 CPU 侧说明：M-bucket workload、kernel/backend selector、策略 selector 与 cost model、gate 阈值。 |

## 第 3 步：已有成果（docs/ 按阶段）

每个阶段建议“先读 `.md` 报告、再对照 `.json` 原始数据”。

### 阶段 0：实机拓扑

| 文件 | 作用 |
|---|---|
| `docs/Q-TopoMoE_8x5090实机拓扑评估与首轮实验矩阵.md` | 8×5090 实机拓扑评估与首轮实验矩阵设计。 |
| `docs/Q-TopoMoE_gpu111_phase0实测分析.md` | gpu-111 实机测量结果分析。 |
| `docs/q_topomoe_phase0_verify.sh` | Phase 0 验证脚本（GPU/NCCL/环境）。 |

### 阶段 1：BF16 / FP8 服务实测

| 文件 | 作用 |
|---|---|
| `docs/results/phase1_service_baseline.md` | 合并后的 BF16/FP8 服务实测叙述报告。 |
| `docs/Q-TopoMoE_Phase1_statistics_20260804.json` | 统计结论原始数据；按 SHA-256 固定。 |

### 阶段 2：量化、规范化检查点与质量 Gate

| 文件 | 作用 |
|---|---|
| `docs/results/phase2_quantization_quality.md` | W4A16 量化预检结论。 |
| `docs/Q-TopoMoE_Qwen35_compat_baseline_20260805.md` + `Qwen35_cleanroom_pins_20260805.json` + `Qwen35_canonical_*` 系列 | 兼容基线、cleanroom 版本 pin、canonical checkpoint 的 tp1/tp2 gate 与 smoke 记录。 |
| `docs/results/phase2_quantization_quality.md` + `docs/Q-TopoMoE_quality_smoke_*_20260805.json` | 质量 smoke 与 vLLM/SGLang、Marlin/Triton backend 隔离记录。 |
| `docs/results/phase2_quantization_quality.md` + `docs/Q-TopoMoE_Phase2_official_like_v2_*_20260805.json` | 116 条 official-like v2 冻结样本的 BF16 vs W4（Triton）质量对比。 |
| `docs/Q-TopoMoE_Phase2_W4A16_*` 系列 | W4A16 审计、freeze、reblock 与 gate 数据。叙述 failure/load/status 已并入 Phase 2 结果。 |
| `docs/Q-TopoMoE_Phase2_WikiText_calibration_manifest.json` | WikiText 校准集机器可读 manifest；叙述已并入 Phase 2 结果。 |

### 阶段 3：route trace 采集与漂移分析

| 文件 | 作用 |
|---|---|
| `docs/results/phase3_route_and_official_eval.md` | 全量 trace 采集与漂移分析报告（Jaccard/flip/相关/CV）。 |
| `docs/Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json` | 全量漂移原始数据（40 层逐层指标）。 |
| `traces/manifests/bf16_full_capture_manifest.json` / `w4a16_full_capture_manifest.json` | 两套全量采集的协议与文件 SHA-256 钉住记录。 |

### 阶段 3c：full-set 官方协议评测

| 文件 | 作用 |
|---|---|
| `docs/results/phase3_route_and_official_eval.md` | 官方协议核实（MMLU-Pro/C-Eval harness 源码）、冻结结果、计时 pilot 与外推。 |
| `docs/Q-TopoMoE_Phase3c_pilot_*_20260806.json` | 官方协议 vs thinking 模式两轮 pilot 的计时/输出统计。 |
| `configs/evaluation/full_set_official_protocol_v1.manifest.json` | 官方协议 full-set（24,374 条）的 SHA-256 与协议参数。 |

### 阶段 4/8

| 文件 | 作用 |
|---|---|
| `docs/Q-TopoMoE_Phase4_Phase8_offline_tasks.md` | 当前可离线执行的 Phase 4/8 任务与命令。 |
| `docs/Q-TopoMoE_Phase8_replay_20260804.json` | 策略回放结果（当前为 blocked_missing_kernel_measurements 预期状态）。 |

## 第 4 步：环境与配置

| 文件 | 作用 |
|---|---|
| `env/activate.sh` | 设置项目/CUDA/NCCL 环境变量（不占 GPU、不启动服务）。 |
| `env/project.env` | 项目级路径与变量定义（被 activate.sh 引用）。 |
| `env/qwen35_cleanroom.env` | cleanroom 的 vLLM/SGLang commit 与 venv 路径。 |
| `env/local.env.example` | 本机个性化配置模板。 |
| `env/check_env.sh` | 环境检查（目录、CUDA、依赖、模型）。 |
| `env/requirements-lock/quant.txt` | 量化/评测环境依赖锁。 |
| `Makefile` | 快捷入口：`make check`（校验）、`make snapshot`（环境快照）、`make tree`（目录树）。 |
| `scripts/bootstrap_qwen35_cleanroom.sh` | 从源码构建 vLLM/SGLang cleanroom venv。 |
| `scripts/create_quant_env.sh` | 创建量化评测 venv。 |
| `scripts/snapshot_env.sh` | 保存环境快照到 artifacts。 |
| `scripts/validate_activation.sh` | 校验激活后的环境。 |
| `configs/evaluation/README.md` | 评测输入（frozen jsonl/manifest）的生成与语义说明。 |
| `configs/models/registry.yaml` / `configs/experiments/*.yaml` / `configs/workloads/*.yaml` / `configs/kernels/*.json` / `configs/strategies/*.json` / `configs/communication/nccl_cost_db.json` | 模型、实验、workload、kernel 计划/DB、策略候选、通信成本等配置。 |

## 第 5 步：复现主线（核心代码）

### 5.1 一键 gate（服务 + 验收 + 质量）

| 文件 | 作用 |
|---|---|
| `scripts/gate_qwen35_checkpoint.sh` | **核心入口**：启动 vLLM/SGLang → checkpoint SHA → acceptance → smoke → quality_eval → 输出 gate_status.json。支持 BACKEND/MODEL_PATH/GPU_IDS/TP_SIZE/QUALITY_MANIFEST 等环境变量。 |
| `serving/acceptance.sh` | 服务验收：health、model discovery、completion、metrics。 |
| `serving/start_server.sh` | 手动启动服务（gate 脚本之外的单次启动）。 |
| `clients/smoke.py` | 并发冒烟客户端（吞吐/延迟）。 |
| `clients/quality_eval.py` | 质量评测客户端：并发请求、答案提取、评分、计时统计、summary。 |
| `evaluation/compare_quality.py` | 两个 summary 的质量对比（drop points + gate）。 |

### 5.2 评测输入冻结

| 文件 | 作用 |
|---|---|
| `evaluation/fetch_official_protocol_assets.py` | 下载官方协议原始资产（MMLU-Pro/C-Eval parquet），钉 revision。 |
| `evaluation/freeze_quality_sets.py` | 生成确定性回归集（quality_smoke_v1 / quality_formal_v1）。 |
| `evaluation/freeze_official_like_smoke.py` | 生成 116 条 official-like sampled 协议。 |
| `evaluation/freeze_full_set.py` | 冻结旧版 full-set（thinking 模式、32768 上限）。 |
| `evaluation/freeze_full_set_official.py` | **当前版本**：冻结官方协议 full-set（MMLU-Pro 5-shot CoT temp0 / C-Eval answer-only，thinking 关）。 |
| `evaluation/slice_pilot.py` | 从 full-set 按 category/subject 分层切计时 pilot。 |

### 5.3 route trace

| 文件 | 作用 |
|---|---|
| `traces/capture_routes.py` | 通过 vLLM `enable_return_routed_experts` 采集逐 token expert ID，输出 npy + manifest + histogram。 |
| `traces/README.md` | 采集输出格式与边界说明（不含 router 概率/KL）。 |
| `analysis/route_drift.py` | BF16 vs W4 漂移分析：Jaccard/flip/相关/CV/M-bucket delta。 |

## 第 6 步：专项深入

### 6.1 量化与 checkpoint

| 文件 | 作用 |
|---|---|
| `quantization/quant_preflight.py` | 量化前预检（权重覆盖、专家审计）。 |
| `quantization/quantize_w4a16.py` | W4A16 量化入口（压缩格式与 Marlin/Triton 兼容性说明）。 |
| `scripts/canonicalize_qwen35_text_checkpoint.py` | 文本 checkpoint 规范化（model.language_model.* → model.*）。 |
| `scripts/audit_w4a16.py` | W4A16 权重/格式审计。 |
| `scripts/freeze_checkpoint.py` / `scripts/freeze_baseline.sh` / `scripts/fetch_wikitext_calibration.py` | 冻结 checkpoint/baseline、拉取校准数据。 |

### 6.2 拓扑与通信

| 文件 | 作用 |
|---|---|
| `topology/collect_hardware.sh` | 硬件信息采集。 |
| `topology/gpu_peer_bf16.py` | GPU peer 带宽测量。 |
| `topology/run_nccl_formal.sh` / `topology/parse_nccl_formal.py` | NCCL 正式矩阵执行与解析。 |
| `scripts/build_nccl_cost_db.py` / `scripts/aggregate_phase1.py` / `scripts/analyze_phase1.py` / `scripts/phase1_matrix.sh` | Phase 1 矩阵执行、聚合、分析，通信成本入库。 |

### 6.3 阶段 4/8 选择器

| 文件 | 作用 |
|---|---|
| `phase4/workload/generate_m_buckets.py` | 生成 90 个确定性 M-bucket workload 用例。 |
| `selector/kernel_db.py` | kernel 数据库（仅 measured+valid 行可被选择）。 |
| `selector/backend_selector.py` | 按 M-bucket/precision 选择 kernel/backend。 |
| `selector/strategy_selector.py` | Phase 8 策略选择与 cost model（compute+comm+imbalance+migration）。 |
| `scripts/plan_phase4_kernel_benchmark.py` / `scripts/replay_phase8.py` | kernel 基准计划生成与 Phase 8 回放。 |

## 第 7 步：测试验证

| 文件 | 作用 |
|---|---|
| `tests/test_quality_eval.py` | quality_eval 答案提取/评分单测。 |
| `tests/test_canonicalize_qwen35_checkpoint.py` | checkpoint 规范化单测。 |
| `tests/test_selector_and_workload.py` | selector 与 M-bucket workload 单测。 |
| `scripts/validate_configs.py` | 配置文件 schema 校验（`make check` 调用）。 |

## 推荐的完整复现路径（最小动作集）

```bash
# 1. 环境
source env/activate.sh && make check
# 2. 服务质量 gate（BF16 示例）
BACKEND=vllm MODEL_PATH=<bf16-snapshot> GPU_IDS=0,1,2,3 TP_SIZE=4 \
  PORT=31350 SERVED_NAME=qtopomoe-gate OUT_DIR=/data/models/test/qtopomoe_gate \
  QUALITY_MANIFEST=configs/evaluation/official_like_smoke_v2.jsonl \
  QUALITY_CONCURRENCY=4 QUALITY_TIMEOUT=3600 bash scripts/gate_qwen35_checkpoint.sh
# 3. full-set 官方协议冻结（资产已下载后）
python evaluation/freeze_full_set_official.py --raw-root <raw> --output <out.jsonl> \
  --tokenizer-a <bf16> --tokenizer-b <w4>
# 4. trace 采集与漂移（需要 cleanroom vLLM）
python traces/capture_routes.py ... && python analysis/route_drift.py ...
```

> 注意：大体积 JSONL/parquet/npy 不入库，复现前需用
> `evaluation/fetch_official_protocol_assets.py` 重新获取并核对 manifest 哈希。
