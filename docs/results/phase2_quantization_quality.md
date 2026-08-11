# 阶段 2：量化、规范化检查点与质量 Gate

> 本文件将阶段 2 的历史报告合并为中文说明。路径、命令、框架版本、模型键名、哈希和协议中的固定英文文本保持原样，以保证结果可复现；机器可读产物不做翻译或改名。

## 源文件完整性

| 原始文件 | UTF-8 字节数 | Git blob SHA-256 |
|---|---:|---|
| `docs/Q-TopoMoE_Phase2_NVFP4_gate_20260807.md` | 6407 | `6E3B46BCBE41BC9431886AA992B6D542F30677950BA703B743457A0F00C88617` |
| `docs/Q-TopoMoE_Phase2_official_like_v2_BF16_W4_results_20260805.md` | 4988 | `76CA73A02897F78F1C204355A46CB5CEB9B5DA5199A7F037234E25D2A485BCFA` |
| `docs/Q-TopoMoE_Phase2_quality_smoke_and_backend_gate_20260805.md` | 2865 | `C8819C316AD364079BED2F8923785B4F7BB4D3B6A5D046C84A4B7233EB19D581` |
| `docs/Q-TopoMoE_Phase2_W4A16_failure_log.md` | 870 | `4F1C67E987A5F64F6268BBF072D3C6D61C162BF92C19BA6BBDA3EEB30B727358` |
| `docs/Q-TopoMoE_Phase2_W4A16_load_gate_20260805.md` | 947 | `2F283AF7C423A549A06C64AD9A73C645F3C8046E87ECFADB7BEA5300052CACE6` |
| `docs/Q-TopoMoE_Phase2_W4A16_run_status.md` | 1212 | `6305D7D4F17D35E1654B73B5EDCEF43BF3FBAE9B81AC4F5A9779A1EE4864CF11` |
| `docs/Q-TopoMoE_Phase2_WikiText_calibration_report.md` | 3599 | `C2E3F2E716D46E314504026F6A9BDAE75AC2165CC9386DB9A4F51524C55368E7` |
| `docs/Q-TopoMoE_Phase2_量化预检报告.md` | 3585 | `C453A1B644C100CB55B7119AC0E6CB349E594C038E54FAE42B7ADC7ABEA17C72` |

## 1. 官方 NVFP4 检查点

官方来源为 `RedHatAI/Qwen3.6-35B-A3B-NVFP4`（通过 ModelScope 镜像下载），格式为 compressed-tensors 的 `nvfp4-pack-quantized`。这是 W4A4：权重和激活均使用 FP4，group size 为 16，scale dtype 为 `fp8e4m3`。模型为 Qwen3.5 MoE 同构结构，共 40 层、256 个 expert、约 3B active 参数；本地 `model.safetensors` 大小为 22.46 GiB，目录为 `/data/models/test/redhatai_qwen36_nvfp4`。

静态覆盖审计结果为 **pass**：routed expert 的 packed weight、weight scale、weight global scale、input global scale 均为 30,720；40 层、256 expert；down/gate/up 三类 projection 各 10,240；linear_attn 未被量化。审计结果写入 `Q-TopoMoE_Phase2_NVFP4_audit_20260807.json`。

## 2. 官方 NVFP4 真实加载与质量

在 vLLM cleanroom `33c50587d` 中，TP1（GPU0）和 TP2（GPU0-1）均通过 health、`/v1/models`、completion 以及 32 请求 smoke。vLLM 自动选择 `VLLM_CUTLASS`，TP1 加载显存 21.88 GiB，TP2 为每卡 11.04 GiB；没有回退到 EMULATION，说明 SM120 上存在真实 NVFP4 MoE kernel。

冻结的 official-like v2 共 116 条样本，BF16 与官方 NVFP4 结果如下：

| 数据集 | BF16 | 官方 NVFP4 | 差值 |
|---|---:|---:|---:|
| C-Eval（52） | 49/52 = 94.23% | 48/52 = 92.31% | -1.92pp |
| MMLU-Pro（64） | 59/64 = 92.19% | 59/64 = 92.19% | 0 |
| 合计（116） | 108/116 = 93.10% | 107/116 = 92.24% | -0.86pp |

weight coverage 无警告、smoke 32/32、quality 116/116 且无失败，因此 sampled Gate 标记为 **accepted**。C-Eval 只有 1 题差异，-1.92pp 略超过预注册的 1.5pp 单项门槛；MMLU-Pro 持平、总体仅下降 0.86pp。该结果只能作为 sampled 协议的边界证据，full-set 结果仍是最终判定依据。

## 3. 自生成 NVFP4 配方与服务 Gate

自生成检查点严格使用 Runbook 6.3：256 条 UltraChat `train_sft`、最大长度 4096、`targets="Linear"`、`moe_calibrate_all_experts=True`，量化格式为 compressed-tensors NVFP4。随后由 `scripts/canonicalize_qwen35_text_checkpoint.py` 把 `model.language_model.*` 映射为 text-only 架构需要的 `model.*`；共 123,972 个键完成 shape、dtype、value 一致性验证。

- 静态覆盖：30,720 个 routed-expert packed/scale/global/input scale 全部存在；40 层、256 expert、down/gate/up 各 10,240；
- TP1、TP2：health、models、completion、metrics、smoke 均通过，backend 为 `VLLM_CUTLASS`；
- TP4 质量服务：smoke 32/32，质量请求 116/116，`failed=0`、`truncated=0`；
- P2P：矩阵通过，未设置 `NCCL_P2P_DISABLE`。

与固定 BF16 r2 对照时，自生成 NVFP4 的 C-Eval 为 49/52 = 94.23%（差 0），MMLU-Pro 为 55/64 = 85.94%（下降 6.25pp），合计 104/116 = 89.66%（下降 3.45pp），因此 Gate **rejected**。8 个正确性翻转均为停止位置变化，没有失败请求或截断；C-Eval 为 2 题丢失、2 题新增，MMLU-Pro 净减少 4 题。

### 3.1 差异诊断

两种检查点的量化覆盖相同（30,720 routed projection，加 120 个 shared projection 和 40 个 attention projection），且全部 30,720 个 `weight_global_scale` 相同；但 `input_global_scale` 仅 9,226/30,720 相同，ratio 的 p05/p50/p95 为 0.9098/1.0000/1.0769，相关系数为 0.9946。256 条校准 token 序列完全一致，因此更可能是 LLM Compressor / compressed-tensors 版本、MoE linearization、pipeline 或执行顺序造成的激活输入 scale 校准差异。不能通过改 scorer 或借用外部 scale“修复”。

## 4. W4A16 结果与框架适配

固定 BF16 与 W4A16 Triton 的 sampled 结果为：BF16 C-Eval 49/52 = 94.23%、MMLU-Pro 59/64 = 92.19%；W4 C-Eval 50/52 = 96.15%、MMLU-Pro 58/64 = 90.63%。两项均在 5 个百分点 smoke 阈值内，Gate 为 pass。

质量 smoke/backend Gate 共 164 行：BF16 vLLM Triton 为 GSM8K 96.88%、C-Eval 86.54%、MMLU 42.19%；W4 Marlin 为 59.38%、36.54%、20.31%，未通过 10pp 阈值；W4 Triton 为 93.75%、82.69%、37.50%；SGLang W4 为 93.75%、78.85%、40.63%。Marlin 的严重回退说明该路径不能作为本模型结论；Triton 的下降分别为 3.13、3.85、4.69pp。证据目录位于 `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/`。

W4 首次量化失败是 Accelerate hook 处理触发的 `AttributeError: functools.partial`（没有 `__func__`），未生成检查点；提交 `0f473f1` 改为递归移除 hook 后修复。另一次 W4 load Gate 因 vLLM `Qwen3_5MoeTextConfig` 与 `Qwen3_5MoeConfig` 不一致、SGLang 找不到兼容的 `Qwen3_5MoeForCausalLM` 而失败，属于 loader/architecture 适配问题，不是 OOM。旧运行 PID 为 3851061，曾在 41 个校准阶段的第 2 阶段约 28%，当时尚未写出检查点。

## 5. WikiText 校准输入与预检

数据集为 `Salesforce/wikitext`、subset `wikitext-2-raw-v1`、split `train`，revision `b08601e...`，源 parquet SHA-256 为 `e83889...`。原始数据 36,718 行，seed=42 固定抽取 256 条记录，目标 2048 token，包装后长度范围 2067–2544。校准 JSONL SHA-256 为 `142e9aa...`；tokenizer config、tokenizer、vocab、merges 与 chat template 哈希分别为 `5186...`、`5f9e...`、`ce99...`、`a9d...`、`e84f...`。原始数据与输出目录仍以服务器 manifest 为准；本报告不宣称已经产生可用校准输出。

预检环境为 Python 3.12.3、torch 2.11+cu130、Transformers 5.10.1、accelerate 1.13.0、datasets 5.0、safetensors 0.8、llmcompressor 0.12.0.1、compressed-tensors 0.17.1、auto-round 0.13.0。预检结果为 **BLOCKED**：未设置 `QTOPOMOE_CALIBRATION_JSONL`，且 64 条 smoke 数据不能替代 256 条校准集。

## 6. 复现入口

评测输入见 [`configs/evaluation/official_like_smoke_v2.jsonl`](../../configs/evaluation/official_like_smoke_v2.jsonl) 与 [`configs/evaluation/official_like_smoke_v2.manifest.json`](../../configs/evaluation/official_like_smoke_v2.manifest.json)；NVFP4 审计 JSON 位于 `docs/Q-TopoMoE_Phase2_NVFP4_audit_20260807.json`。服务启动、质量抽取和 Gate 判定应严格使用仓库脚本与 manifest，不要手工改写结果。
