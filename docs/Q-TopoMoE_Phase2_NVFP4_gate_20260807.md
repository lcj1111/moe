# Q-TopoMoE Phase 2 NVFP4：官方与自生成 checkpoint 验证

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

## 5. 自生成 NVFP4（2026-08-09）

自生成 checkpoint 使用 Runbook 6.3 固定配方：256 条 UltraChat
`train_sft`、最大长度 4096、`targets="Linear"`、
`moe_calibrate_all_experts=True`、compressed-tensors NVFP4。量化输出先经
`scripts/canonicalize_qwen35_text_checkpoint.py` 将
`model.language_model.*` 映射为 text-only 架构所需的 `model.*`；123972
个键完成 shape/dtype/value 一致性验证。

### 5.1 覆盖与服务 Gate

- 静态覆盖：pass；30720 个 routed-expert packed/scale/global/input scale
  完整，40 层、256 experts、down/gate/up 各 10240；
- TP1：health/models/completion/metrics 全通过，smoke c1 1/1、c32 32/32，
  backend `VLLM_CUTLASS`；
- TP2：同样全通过，smoke 32/32，backend `VLLM_CUTLASS`；
- TP4 质量服务：smoke 32/32；质量请求 116/116，failed=0、truncated=0；
- P2P 为共同控制条件：8x8 read/write peer matrix 除对角线外均为 `OK`，
  进程未设置 `NCCL_P2P_DISABLE`。因此质量差异不归因于 P2P 开关。

### 5.2 质量结果与独立判定

基线必须使用修复后的 `bf16_vllm_tp4_r2`（116/116、failed=0），不能使用
早期有 2 条请求失败的 `bf16_vllm_tp4`：

| 项 | BF16 r2 | 自生成 NVFP4 | 差 |
|---|---:|---:|---:|
| C-Eval（52） | 49/52 = 94.23% | 49/52 = 94.23% | 0 |
| MMLU-Pro（64） | 59/64 = 92.19% | 55/64 = 85.94% | -6.25pp |
| 合计（116） | 108/116 = 93.10% | 104/116 = 89.66% | **-3.45pp** |

服务脚本生成的 `gate_status=accepted` 仅表示请求完整、服务健康；按 Runbook
预注册质量门槛（NVFP4 相对 BF16 下降不超过 1.5pp），自生成 checkpoint
的质量 Gate 为 **rejected**。8 个相对 BF16 的 correctness flip 全部
`finish_reason=stop`、无截断、无请求错误；C-Eval 两失两得，MMLU-Pro
净损失 4 题，因此不是答案抽取或服务异常。

### 5.3 诊断

与已通过的 RedHatAI checkpoint 对比：两者 quantized coverage 完全相同
（30720 routed experts + 120 shared-expert projections + 40 self-attention
projections），30720 个 expert `weight_global_scale` 全部相等；差异集中在
`input_global_scale`，仅 9226/30720 完全相等，selfgen/RedHat 比值
p05/p50/p95 = 0.9098/1.0000/1.0769，相关系数 0.9946。对 256 条校准样本
复算后，官方 processor 路径与本项目 tokenizer/string 路径的截断后 token
序列 256/256 完全一致。

这排除了覆盖、源权重量化、校准样本和 tokenization 不一致；现有证据把
退化定位到 activation input scale 的校准执行差异（LLM Compressor/
compressed-tensors 版本、MoE linearization/独立 pipeline 或样本执行次序）。
在没有新的冻结配方与全套复测前，不应通过改评分器或移植外部 scale
“修复”该 Gate。

## 6. 结论与下一步

1. 官方 NVFP4 checkpoint 静态覆盖完整、真实加载通过、质量总体达标，
   满足 Runbook 6.3"先验证官方 checkpoint"的前置条件。
2. 自生成纯 NVFP4 的覆盖和 TP1/TP2/TP4 服务 Gate 全通过，但总体质量
   -3.45pp，超过 1.5pp 门槛，按 Runbook 退出正式系统主线并保留失败证据。
3. 后续实验继续使用已通过的 RedHatAI NVFP4 checkpoint；下一步采集其
   route trace 与真实 M-bucket。自生成改进分支只允许以冻结的新校准执行
   配方重新量化、重新跑完整 Gate，不覆盖当前 v1 证据。

原始数据：
[audit](Q-TopoMoE_Phase2_NVFP4_audit_20260807.json) /
[official quality summary](Q-TopoMoE_Phase2_NVFP4_official_like_v2_summary_20260807.json) /
[selfgen gate](Q-TopoMoE_Phase2_NVFP4_selfgen_gate_20260809.json) /
[selfgen quality summary](Q-TopoMoE_Phase2_NVFP4_selfgen_official_like_v2_summary_20260809.json)。
