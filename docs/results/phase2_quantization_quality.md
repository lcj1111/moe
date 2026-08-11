# Phase 2 — quantization, canonical checkpoints and quality gates

> Consolidated from the dated reports listed below. Source content is retained; only trailing whitespace was normalized. SHA-256 values are computed from the UTF-8 Git blob (LF-normalized); machine-readable artifacts keep their original paths for reproducibility.

## Source integrity

| Original file | UTF-8 bytes | SHA-256 of Git blob |
|---|---:|---|
| `docs/Q-TopoMoE_Phase2_NVFP4_gate_20260807.md` | 6407 | `6E3B46BCBE41BC9431886AA992B6D542F30677950BA703B743457A0F00C88617` |
| `docs/Q-TopoMoE_Phase2_official_like_v2_BF16_W4_results_20260805.md` | 4988 | `76CA73A02897F78F1C204355A46CB5CEB9B5DA5199A7F037234E25D2A485BCFA` |
| `docs/Q-TopoMoE_Phase2_quality_smoke_and_backend_gate_20260805.md` | 2865 | `C8819C316AD364079BED2F8923785B4F7BB4D3B6A5D046C84A4B7233EB19D581` |
| `docs/Q-TopoMoE_Phase2_W4A16_failure_log.md` | 870 | `4F1C67E987A5F64F6268BBF072D3C6D61C162BF92C19BA6BBDA3EEB30B727358` |
| `docs/Q-TopoMoE_Phase2_W4A16_load_gate_20260805.md` | 947 | `2F283AF7C423A549A06C64AD9A73C645F3C8046E87ECFADB7BEA5300052CACE6` |
| `docs/Q-TopoMoE_Phase2_W4A16_run_status.md` | 1212 | `6305D7D4F17D35E1654B73B5EDCEF43BF3FBAE9B81AC4F5A9779A1EE4864CF11` |
| `docs/Q-TopoMoE_Phase2_WikiText_calibration_report.md` | 3599 | `C2E3F2E716D46E314504026F6A9BDAE75AC2165CC9386DB9A4F51524C55368E7` |
| `docs/Q-TopoMoE_Phase2_量化预检报告.md` | 3585 | `C453A1B644C100CB55B7119AC0E6CB349E594C038E54FAE42B7ADC7ABEA17C72` |

---

## Source: `docs/Q-TopoMoE_Phase2_NVFP4_gate_20260807.md`

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

---

## Source: `docs/Q-TopoMoE_Phase2_official_like_v2_BF16_W4_results_20260805.md`

# Q-TopoMoE Phase 2 official-like v2：BF16 与 W4A16（Triton MoE）质量对比结果

> 生成日期：2026-08-05（Asia/Shanghai）
>
> 性质：protocol-aligned **sampled** reproduction（116 条冻结样本），不是官方 full-set 复现。
> 冻结 manifest：`configs/evaluation/official_like_smoke_v2.jsonl`（manifest SHA-256
> `549c1e3b3e131e8ff586dbf17ae666fb8468b75b57362e206cfbce2a4fcb6d0f`）。

## 1. 前置修复记录

BF16 official-like v2 首次运行（`bf16_vllm_tp4`，17:24 结束）未通过：

- `gate_status.status = quality_eval_failed`；
- `requested=116, completed=114, failed=2, truncated=0`；
- 失败行 `mmlu_pro:test:11397`、`mmlu_pro:test:11774` 均为客户端
  `TimeoutError`（aiohttp `ClientTimeout(total=600)`，`request_ms≈600.6s`），
  服务端无 500/崩溃；根因是这两个超长 thinking 请求的生成时间超过
  默认 600 秒客户端超时（该机单请求约 14.5 tok/s）。

修复（commit `dd7c78e`）：`scripts/gate_qwen35_checkpoint.sh` 新增
`QUALITY_TIMEOUT`（默认 3600s）并透传给 `clients/quality_eval.py --timeout`。
重跑使用同一冻结 manifest/seed，仅超时参数不同。

## 2. BF16 v2 重跑（accepted）

运行目录：
`/data/models/test/qtopomoe_w4a16_runs/official_like_smoke_v2/bf16_vllm_tp4_r2`

服务配置：

| 项 | 值 |
|---|---|
| Backend | vLLM（cleanroom commit `33c50587d`） |
| Model | `Qwen--Qwen3.6-35B-A3B/snapshots/master`（BF16） |
| GPU / TP | 0,1,2,3 / TP4 |
| Port | 31357（仅 localhost） |
| Served name | `qtopomoe-bf16-official-like-v2` |
| MAX_MODEL_LEN / MAX_NUM_SEQS / MEM_FRACTION | 65536 / 4 / 0.90 |
| QUALITY_CONCURRENCY / QUALITY_TIMEOUT | 4 / 3600s |
| VLLM_MOE_BACKEND | auto（BF16 非量化） |

Gate 结果：`status=accepted`，`requested=116, completed=116, failed=0,
truncated=0`，`quality.results.jsonl` 116 行。

审计：`errors=0`、`truncated=0`、`unparsed=0`，finish_reason 全部为
`stop`（无 `length`），无请求到 max_tokens 上限，MMLU-Pro 64/64、
C-Eval 52/52 全部 scored 且预测为合法单选项。

成绩：

| Benchmark | records/scored | correct | accuracy |
|---|---:|---:|---:|
| C-Eval | 52/52 | 49 | 92.31% |
| MMLU-Pro | 64/64 | 59 | 92.19% |

## 3. W4A16 Triton MoE（accepted）

运行目录：
`/data/models/test/qtopomoe_w4a16_runs/official_like_smoke_v2/w4a16_vllm_triton_tp4`

服务配置：

| 项 | 值 |
|---|---|
| Backend | vLLM（cleanroom commit `33c50587d`） |
| Model | `/data/models/test/qtopomoe_w4a16_canonical_text_v1` |
| Checkpoint 校验 | 权重 SHA-256 `0eb2775989321d038ea9534041a6030f536a0d0d2293e7087837390e9738c1d2`，93,093 tensor，`model.language_model.* -> model.*` 唯一映射 |
| GPU / TP | 0,1,2,3 / TP4 |
| Port | 31358（仅 localhost） |
| Served name | `qtopomoe-w4a16-triton-official-like-v2` |
| MAX_MODEL_LEN / MAX_NUM_SEQS / MEM_FRACTION | 65536 / 4 / 0.90 |
| QUALITY_CONCURRENCY / QUALITY_TIMEOUT | 4 / 3600s |
| VLLM_MOE_BACKEND | **triton**（主线强制；MoE Marlin 为已拒绝候选） |

Gate 结果：`status=accepted`，`requested=116, completed=116, failed=0,
truncated=0`，`quality.results.jsonl` 116 行。

审计：`errors=0`、`truncated=0`、`unparsed=0`，finish_reason 全部为
`stop`，无请求到上限，64/64 + 52/52 全部 scored 且预测合法。

成绩：

| Benchmark | records/scored | correct | accuracy |
|---|---:|---:|---:|
| C-Eval | 52/52 | 50 | 96.15% |
| MMLU-Pro | 64/64 | 58 | 90.63% |

## 4. BF16 vs W4 对比（gate_pass=true）

对比脚本：`evaluation/compare_quality.py`，输出：
`docs/Q-TopoMoE_Phase2_official_like_v2_BF16_vs_W4_compare_20260805.json`

| Benchmark | records | BF16 acc | W4 acc | drop points | 阈值 | pass |
|---|---:|---:|---:|---:|---:|---|
| C-Eval | 52 | 92.31% | 96.15% | -1.92 | 5.0 | true |
| MMLU-Pro | 64 | 92.19% | 90.63% | +1.56 | 5.0 | true |

`gate_pass=true`。HumanEval 在隔离代码执行可用前排除。

## 5. 结论

1. canonical W4A16 + vLLM Triton MoE 在 official-like v2 冻结样本上与 BF16
   质量基本持平（C-Eval 反高 1.92 点、MMLU-Pro 低 1.56 点），远优于此前
   Marlin MoE 的严重退化（GSM8K 59.38% vs Triton 93.75%），验证了
   backend 数值隔离决策的正确性。
2. 两边 records/scored 完全一致（52/52、64/64）、truncated=0，
   满足比较前置条件。
3. 5 points 为 sampled smoke 的暂定 Gate 阈值，不是论文最终阈值；
   1.5–2 点量级差异落在 sampling noise 范围内，不能据此宣称 W4 无损或
   有损，需要正式重复（≥5 次独立运行 + 置信区间）后下结论。

## 6. 边界与后续

- 这是 116 条 sampled 结果，不是官方 full-set 复现；不得将 92% 级别数字
  当作官方分数。
- 后续主线：route trace / expert token histogram 采集 → 真实 M 桶驱动
  Phase 4 kernel 基准 → 填充 kernel DB → Phase 8 replay →
  关键候选 5 次独立正式重复 → 论文实验章节。

---

## Source: `docs/Q-TopoMoE_Phase2_quality_smoke_and_backend_gate_20260805.md`

# Q-TopoMoE Phase 2 quality smoke and backend gate

Date: 2026-08-05 CST

## Frozen controls

- Quality set: `/data/models/test/qtopomoe_quality/frozen/quality_smoke_v1.jsonl`
- Frozen rows: 164 (GSM8K 32, MMLU-Pro 64, C-Eval 52, HumanEval 16)
- Automatically scored rows: 148. HumanEval is not executed outside an isolated sandbox.
- All runs use seed 42, concurrency 8, TP4 on GPUs 0-3, max model length 4096, and the exact vLLM cleanroom commit `33c50587d2679ba9bacc2a51ae19901f7eb3a129` unless stated otherwise.

## Results

| Runtime | MoE backend | GSM8K | C-Eval | MMLU-Pro | Requests |
|---|---|---:|---:|---:|---:|
| BF16 vLLM | Triton unquantized | 96.88% | 86.54% | 42.19% | 148/148 |
| W4A16 vLLM | Marlin | 59.38% | 36.54% | 20.31% | 148/148 |
| W4A16 vLLM | Triton WNA16 | 93.75% | 82.69% | 37.50% | 148/148 |
| W4A16 SGLang | native WNA16 Marlin | 93.75% | 78.85% | 40.63% | 148/148 |

The W4A16 vLLM Marlin result fails the 10-point smoke threshold on all three
benchmarks. The same checkpoint passes when only the vLLM MoE backend is
changed to `triton`: drops versus vLLM BF16 are 3.13, 3.85, and 4.69 points.

This isolates the severe regression to the current vLLM Marlin MoE path for
this Qwen3.5 MoE / compressed-tensors W4A16 / SM120 / TP4 combination. Dense
W4A16 Linear layers remain on `MarlinLinearKernel`; only MoE experts are forced
to `TritonWNA16Experts` with `--moe-backend triton`.

## Rejected or diagnostic paths

- Ordinary Transformers loading is not a valid checkpoint oracle here. It
  constructs fused expert parameters before llmcompressor can linearize the
  model, so the saved per-expert compressed keys are unexpected and fused
  expert parameters are missing.
- SGLang BF16 TP4 with the original conditional-generation checkpoint failed
  in multimodal CUDA IPC because peer access is not supported between the
  selected PCIe GPUs. This does not invalidate the SGLang W4 text-only run.
- `--language-only` was removed from the monolithic SGLang Gate. In this
  SGLang revision it enables encoder-disaggregation semantics; at TP4 the
  Qwen3.5 text class is rejected on that path.

## Evidence directories

- BF16 vLLM: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/bf16_vllm_tp4`
- W4 vLLM Marlin: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/w4a16_vllm_tp4`
- W4 vLLM Triton: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/w4a16_vllm_triton_tp4`
- W4 SGLang: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/w4a16_sglang_tp4_retry1`
- Comparisons: `/data/models/test/qtopomoe_w4a16_runs/quality_smoke/compare_bf16_vs_w4_{marlin,triton}.json`

## Gate decision

Proceed to the formal frozen quality set with vLLM BF16 as reference and vLLM
W4A16 using `VLLM_MOE_BACKEND=triton`. Keep vLLM Marlin as a rejected backend
candidate and retain SGLang W4 as independent checkpoint-quality evidence.

---

## Source: `docs/Q-TopoMoE_Phase2_W4A16_failure_log.md`

# W4A16 execution log

The first real W4A16 attempt used 256 frozen calibration records and loaded the
35B checkpoint, but failed before calibration at compressed-tensors hook
initialization:

```text
AttributeError: 'functools.partial' object has no attribute '__func__'
```

Cause: `device_map="auto"` in Transformers/Accelerate installs partial forward
hooks, while compressed-tensors 0.17.1 expects a bound method when wrapping a
Linear module. No checkpoint was written and all GPU memory returned to idle.

The entrypoint now removes Accelerate hooks recursively after loading and lets
the llmcompressor calibration pipeline install its own compressed-tensors
offload hooks. The fix is in commit `0f473f1`; the next retry additionally
includes the recursive hook removal and is expected to re-run the same frozen
input without changing its hash or sample indices.

---

## Source: `docs/Q-TopoMoE_Phase2_W4A16_load_gate_20260805.md`

# W4A16 real-load Gate (2026-08-05)

Static coverage passed: the checkpoint contains 40 layers × 256 experts × 3
expert projections = 30,720 packed expert linears and 30,720 scales, with
`linear_attn` excluded. The checkpoint is `compressed-tensors` W4A16 and is
20G on disk.

The real service Gate did not pass. vLLM 0.26.0 failed before allocation with a
Qwen3.5 MoE config type mismatch (`Qwen3_5MoeTextConfig` versus the vLLM
`Qwen3_5MoeConfig` expected by its multimodal renderer). SGLang 0.5.16 failed
before serving because it has no compatible `Qwen3_5MoeForCausalLM`
implementation. These are loader/architecture failures, not OOM evidence.

Consequently `/health`, model discovery, completion, metrics, concurrency
smoke, quality comparison and route capture are not yet valid for W4A16. TP,
EP, EPLB and Phase 8 real strategy selection remain blocked until a supported
loader or an explicitly reviewed model-adapter fix is validated.

---

## Source: `docs/Q-TopoMoE_Phase2_W4A16_run_status.md`

# Q-TopoMoE Phase 2 W4A16 run status

Updated: 2026-08-04 16:44 CST
Status: RUNNING (not a completed quantization result)

The retry uses the frozen input without modification:

- model: `/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`
- calibration: `/data/models/test/qtopomoe_calibration/wikitext2_raw_train_256.jsonl`
- calibration SHA-256: `142e9aa4a9821caec5fb10537485d7cf99cbc1bdb7336bf1cb8ab65b5c5e76cf`
- samples / max sequence length: `256 / 4096`
- command PID: `3851061`
- log: `/data/models/test/qtopomoe_quant_runs/w4a16_20260804_retry2.log`
- output target: `/data/models/test/qtopomoe_w4a16`

Observed state at update time:

- llmcompressor initialized `GPTQModifier` and selected `SequentialPipeline`;
- model calibration reached subgraph `(2/41)` at approximately 28%;
- no OOM or traceback after the recursive Accelerate-hook fix;
- no W4A16 checkpoint has been written yet.

Monitor with:

```bash
ps -p 3851061 -o pid,stat,etime,%cpu,%mem,cmd
tail -f /data/models/test/qtopomoe_quant_runs/w4a16_20260804_retry2.log
```

Completion still requires checkpoint existence, the 30,720-expert-scale
coverage audit, and real SGLang/vLLM load and service validation.

---

## Source: `docs/Q-TopoMoE_Phase2_WikiText_calibration_report.md`

# Q-TopoMoE Phase 2: Frozen Open Calibration Set

Date: 2026-08-04
Host: gpu-111
Status: calibration input PASS; W4A16 execution not started yet

## Candidate assessment

The selected source is `Salesforce/wikitext`, subset `wikitext-2-raw-v1`,
split `train`. The Hub page documents the dataset as Wikipedia-derived
language-modeling text and lists CC BY-SA/GFDL licensing. A fixed Hub commit
is used rather than a mutable `main` pointer. WikiText is general English
text, not an instruction/chat benchmark; it is suitable for weight and MoE
expert calibration, but it must not be used to claim chat-quality gains.

This is more controllable for the present reproduction than C4: C4 is ODC-BY,
is derived from Common Crawl, and is many terabytes in the Hub card. The
selection is therefore feasible, small enough to archive locally, and easy to
rebuild. License obligations still apply to any redistribution.

## Frozen source and preprocessing

| Field | Value |
|---|---|
| Dataset repository | `Salesforce/wikitext` |
| Revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Config / split | `wikitext-2-raw-v1` / `train` |
| Source parquet SHA-256 | `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` |
| Source rows | 36,718 |
| Selection seed | 42 |
| Records | 256 |
| Target tokens per record | 2,048 before chat wrapping |
| Chat-wrapped token range | 2,067–2,544 |
| Calibration JSONL SHA-256 | `142e9aa4a9821caec5fb10537485d7cf99cbc1bdb7336bf1cb8ab65b5c5e76cf` |

The generator shuffles source row IDs with `random.Random(42)`, packs real
rows until the target token budget is reached, and stores the exact source row
IDs in every record. It never duplicates the 64-record smoke seed.

## Tokenizer and template freeze

Tokenizer root:

`/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`

| File / value | SHA-256 |
|---|---|
| `tokenizer_config.json` | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `vocab.json` | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `merges.txt` | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| Qwen3.6 chat template | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |

The full sample-index list and template text are tracked in
`docs/Q-TopoMoE_Phase2_WikiText_calibration_manifest.json`.

## Server locations and checks

```text
/data/models/test/qtopomoe_wikitext/wikitext-2-raw-v1-train.parquet
/data/models/test/qtopomoe_calibration/wikitext2_raw_train_256.jsonl
/data/models/test/qtopomoe_calibration/wikitext2_raw_train_256.manifest.json
```

The formal preflight passed with 256 records, the Qwen3.6 MoE metadata,
8 visible GPUs, and the pinned quantization environment. The raw parquet and
JSONL remain server-side because the repository ignores large/raw data files;
the pinned revision, source hash, preprocessing script, sample indices, and
output hash are tracked so the files can be regenerated.

Next command after final review:

```bash
cd /home/k8s-ops/moe
source env/project.env
/data/models/test/qtopomoe_quant_env/bin/python \
  quantization/quantize_w4a16.py \
  --model "$QTOPOMOE_BF16_MODEL" \
  --calibration "$QTOPOMOE_CALIBRATION_JSONL" \
  --output /data/models/test/qtopomoe_w4a16 \
  --samples 256 --max-seq-length 4096
```

Quantization output is not yet claimed. It still requires the Runbook 6.4
coverage audit and 6.5 real SGLang/vLLM load, completion, metrics, and
concurrency validation.

---

## Source: `docs/Q-TopoMoE_Phase2_量化预检报告.md`

# Q-TopoMoE Phase 2 量化预检报告

日期：2026-08-04
阶段：6.1 量化环境冻结、6.2 W4A16 入口准备
主机：gpu-111（8 × RTX 5090）

## 结论

6.1 已完成并可复现。独立环境位于 `/data/models/test/qtopomoe_quant_env`，其 CUDA 运行时复用了已验收的服务环境 torch，但量化工具包独立安装并锁定。正式 6.2 目前被校准数据门禁阻塞：项目只有 64 条 smoke seed，没有经数据集版本、样本索引、tokenizer revision、chat template 和 SHA-256 冻结的 256 条校准集，因此没有启动 35B PTQ。

这不是失败的量化结果，而是为了保持可复现性而保留的输入门禁。禁止把 64 条 smoke 数据复制填充到 256 条。

## 6.1 实测环境

执行：

```bash
cd /home/k8s-ops/moe
bash scripts/create_quant_env.sh
```

关键输出：

| 项目 | 实测值 |
|---|---|
| Python | 3.12.3 |
| torch | 2.11.0+cu130 |
| Transformers | 5.10.1 |
| accelerate | 1.13.0 |
| datasets | 5.0.0 |
| safetensors | 0.8.0 |
| llmcompressor | 0.12.0.1 |
| compressed-tensors | 0.17.1 |
| auto-round | 0.13.0 |
| CUDA 可用 / 设备数 | True / 8 |
| pip check | 量化栈通过；仅报告共享 SGLang 要求 Transformers 5.12.1 的边界冲突 |

锁文件：`env/requirements-lock/quant.txt`。共享服务环境中的 SGLang 不属于量化环境，不应为了量化而修改。

## 6.2 预检实测

模型快照：

`/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`

预检命令：

```bash
/data/models/test/qtopomoe_quant_env/bin/python \
  quantization/quant_preflight.py \
  --model /home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master \
  --output artifacts/reports/phase2_quant_preflight_formal_20260804.json
```

已验证：`model_type=qwen3_5_moe`、`Qwen3_5MoeForConditionalGeneration`、40 层、256 experts、8 卡可见、配置 SHA-256 与 Phase 1 manifest 一致，以及约 248 GB 模型盘可用空间。

正式预检返回 `BLOCKED`，唯一门禁是 `QTOPOMOE_CALIBRATION_JSONL is not set`。环境-only 预检返回 `PASS`，用于证明工具链本身可用。

另外使用现有 64 条 smoke seed 对正式入口做了守门测试，入口在加载 35B 权重前返回：

```text
refusing to quantize: calibration has 64 records, requires 256
```

因此当前没有产生任何 W4A16 checkpoint，也没有把 smoke 数据重复扩充为校准集。

## 已准备的正式入口

`quantization/quantize_w4a16.py` 已提交，使用 llmcompressor 0.12.0.1 的：

- `load_quantizable_moe(AutoModelForCausalLM)`，将 fused 3-D experts 线性化；
- `GPTQModifier(targets="Linear", scheme="W4A16")`；
- `moe_calibrate_all_experts=True`；
- 排除 `linear_attn`、router/shared-expert gate、embedding、`lm_head`；
- 256 样本、4096 token、`compressed-tensors` 输出和量化 manifest。

拿到冻结校准集后，先执行正式预检，再执行：

```bash
export QTOPOMOE_CALIBRATION_JSONL=/path/to/pinned_calibration.jsonl
/data/models/test/qtopomoe_quant_env/bin/python \
  quantization/quantize_w4a16.py \
  --model /home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master \
  --calibration "$QTOPOMOE_CALIBRATION_JSONL" \
  --output /data/models/test/qtopomoe_w4a16 \
  --samples 256 --max-seq-length 4096
```

量化完成后仍需按 Runbook 6.4/6.5 做 safetensors 静态覆盖审计、TP1 容量门禁、SGLang/VLLM 实际加载、`/health`、`/v1/models`、真实 completion、`/metrics`，以及并发 1/32 的对照基线；当前均未宣称完成。
