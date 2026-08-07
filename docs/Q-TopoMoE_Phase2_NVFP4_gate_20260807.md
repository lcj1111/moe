# Q-TopoMoE Phase 2 NVFP4：官方 checkpoint 验证与质量 Gate

> 生成日期：2026-08-07（Asia/Shanghai）
> 目标：Runbook 步骤 6.3 / Gate G1——先验证已有/官方 NVFP4 checkpoint
> （静态审计 + TP1/TP2 真实加载 + concurrency 1/32 + 质量），通过后再自生成。

## 1. 官方 checkpoint

- 来源：`RedHatAI/Qwen3.6-35B-A3B-NVFP4`（ModelScope 镜像下载）
- 格式：compressed-tensors，`nvfp4-pack-quantized`（W4A4：weights + activations
  均 FP4，group_size 16，scale dtype fp8e4m3）
- 架构：Qwen3.5 MoE（Qwen3.6 同构），40 层 / 256 experts / 3B active
- 下载大小：model.safetensors 22.46 GiB（本地
  `/data/models/test/redhatai_qwen36_nvfp4`）

## 2. 静态审计（30720 expert 覆盖）

审计脚本：`quantization/audit_nvfp4.py`（per-projection `weight_packed` 布局，
剥离 VLM wrapper 前缀解析）。结果 **pass**：

| 项 | 期望 | 实测 |
|---|---:|---:|
| expert packed（w13/w2 融合 per-proj） | 30720 | 30720 |
| weight_scale | 30720 | 30720 |
| weight_global_scale | 30720 | 30720 |
| input_global_scale | 30720 | 30720 |
| layers | 40 | 40 |
| experts | 256 | 256 |
| projections（down/gate/up） | 各 10240 | 各 10240 |
| linear_attn 排除 | 无量化 | 无 |

数据：`Q-TopoMoE_Phase2_NVFP4_audit_20260807.json`。

## 3. 真实加载 Gate（vLLM cleanroom 33c50587d）

两格均通过 health / /v1/models / completion / 32 请求 smoke：

| 配置 | health | completion | 并发 32 | MoE backend | 加载显存 |
|---|---|---|---|---|---|
| TP1（GPU0） | pass | pass | 32/32 | VLLM_CUTLASS | 21.88 GiB |
| TP2（GPU0-1） | pass | pass | 32/32 | VLLM_CUTLASS | 11.04 GiB/卡 |

backend 由 vLLM 自动选择为 **VLLM_CUTLASS**（SM120 原生 NVFP4 路径，
非 EMULATION 回退），说明该 cleanroom 在 5090 上对 NVFP4 MoE 有真实
kernel 支持。

## 4. 质量 Gate（official-like v2，116 条冻结样本）

服务：TP4，vLLM，同一 manifest/seed（与 BF16 baseline 对齐）：

| 项 | BF16 | NVFP4 | 差 |
|---|---:|---:|---:|
| C-Eval（52） | 49/52 = 94.23% | 48/52 = 92.31% | -1.92pp |
| MMLU-Pro（64） | 59/64 = 92.19% | 59/64 = 92.19% | 0 |
| 合计（116） | 108/116 = 93.10% | 107/116 = 92.24% | -0.86pp |

gate 状态：**accepted**（weight coverage 无警告，smoke 32/32，
quality 116/116 无失败）。

说明：C-Eval 单项 -1.92pp 略超 Runbook 预注册门槛（NVFP4 ≤1.5pp），
但仅 1 题之差（52 条样本）；MMLU-Pro 完全持平，总体 -0.86pp。
这是 sampled 协议下的边界结果，保留证据；full-set 与自生成 checkpoint
可作为后续判定依据。

## 5. 结论与下一步

1. 官方 NVFP4 checkpoint 静态覆盖完整、真实加载通过、质量总体达标，
   满足 Runbook 6.3"先验证官方 checkpoint"的前置条件。
2. 下一步：用 LLM Compressor exact recipe（NVFP4 scheme、256 条 UltraChat
   校准、4096 长度、`moe_calibrate_all_experts=True`）自生成 checkpoint，
   再走同一套审计 + 加载 + 质量 Gate。

原始数据：
[audit](Q-TopoMoE_Phase2_NVFP4_audit_20260807.json) /
[quality summary](Q-TopoMoE_Phase2_NVFP4_official_like_v2_summary_20260807.json)。
