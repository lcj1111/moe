# Q-TopoMoE NVFP4 阶段交接文档

> 生成日期：2026-08-09（Asia/Shanghai）
> 用途：承接"NVFP4 部分"（Runbook 步骤 6.3 / Gate G1）的所有已完成工作、
> 未完成项、环境事实与已遇到问题；任何新窗口/接手人读本文件 +
> 服务器 + GitHub 即可无缝继续。

## 1. 任务目标（Runbook 6.3）

1. 先验证已有/官方 NVFP4 checkpoint：静态审计 → TP1 容量 → TP2 →
   health/models/completion/metrics → concurrency 1/32；
2. 通过后用 LLM Compressor exact recipe 自生成 NVFP4（256 条 UltraChat、
   4096 长度、`targets="Linear"`、`moe_calibrate_all_experts=True`，
   输出 compressed-tensors）；
3. 自生成 checkpoint 走同一套审计 + 加载 + 质量 Gate；
4. 归档报告并同步 GitHub。

质量预注册门槛：NVFP4 相对 BF16 ≤1.5 绝对百分点下降。

## 2. 三端仓库状态（归档前快照）

| 端 | 位置 | HEAD | 备注 |
|---|---|---|---|
| 本地 Windows | `C:\Users\29876\Documents\Codex\2026-08-03\new-chat-2\work\moe-push-20260804` | `9f2db1e` + 本状态提交 | 官方与 selfgen Gate 已归档 |
| GitHub | `lcj1111/moe` 分支 `agent/sync-q-topomoe-project` | `1c79c34` | **尚未同步**：HTTPS 443 超时；GitHub 应用写 API 返回 403 |
| 服务器 | `gpu-111:/home/k8s-ops/moe` | `9f2db1e` + 本状态提交 | bundle 快进同步；`.sync/` 为可恢复备份 |

同步方式：本地 commit → bundle → scp 服务器 → ff-only merge 已完成；待
`github.com:443` 恢复后，从本地执行
`git push origin agent/sync-q-topomoe-project`。可恢复 bundle 位于
`outputs/qtopomoe-9f2db1e.bundle`，服务器副本位于
`/home/k8s-ops/moe/.sync/qtopomoe-9f2db1e.bundle`。

## 3. 已完成工作

### 3.1 官方 checkpoint 验证（全部通过）

- 来源：`RedHatAI/Qwen3.6-35B-A3B-NVFP4`，经 ModelScope 下载到
  `/data/models/test/redhatai_qwen36_nvfp4`（model.safetensors 22.46 GiB）。
  格式：compressed-tensors `nvfp4-pack-quantized`（W4A4，group_size 16，
  scale dtype fp8e4m3），Qwen3.5 MoE 同构（40 层 / 256 experts / 3B active）。
- 静态审计：`quantization/audit_nvfp4.py` → **pass**，30720 expert
  packed/scale/global_scale/input_global_scale 全覆盖，40 层、256 专家、
  down/gate/up 各 10240，linear_attn 已排除。
- 真实加载 Gate（vLLM cleanroom `33c50587d`）：
  - TP1（GPU0）：health/models/completion pass，并发 32/32，
    backend = **VLLM_CUTLASS**（SM120 原生 NVFP4 路径，非 EMULATION 回退），
    加载显存 21.88 GiB；
  - TP2（GPU0-1）：同全 pass，并发 32/32，加载 11.04 GiB/卡。
- 质量 Gate（official-like v2，116 条冻结样本，TP4）：
  - C-Eval 48/52 = 92.31%（BF16 94.23%，-1.92pp）；
  - MMLU-Pro 59/64 = 92.19%（BF16 92.19%，持平）；
  - 合计 107/116 = 92.24%（-0.86pp）；gate 状态 **accepted**，
    weight coverage 无警告，116/116 无失败。
  - 边界说明：C-Eval 单项 -1.92pp 略超 1.5pp 门槛（仅 1 题之差），
    已在报告中如实标注；总体达标。
- 报告与数据：`docs/results/phase2_quantization_quality.md`（本地已提交），
  `docs/Q-TopoMoE_Phase2_NVFP4_audit_20260807.json`，
  `docs/Q-TopoMoE_Phase2_NVFP4_official_like_v2_summary_20260807.json`，
  审计脚本 `quantization/audit_nvfp4.py`。

### 3.2 自生成 NVFP4 checkpoint（Gate 已完成，质量 rejected）

- 量化脚本：`quantization/quantize_nvfp4.py`（复用 W4A16 管线，
  recipe = `QuantizationModifier(scheme="NVFP4")`，官方 ignore 列表）。
- 校准数据：`/data/models/test/qtopomoe_calibration/ultrachat_200k_sft_256.jsonl`
  （256 条 UltraChat `train_sft`，Runbook 指定数据源）。
- 输出：`/data/models/test/qtopomoe_nvfp4_canonical_text_v1`（22.4 GiB）。
- 静态审计：**pass**（30720 expert 覆盖，与官方一致）。
- 键名问题与修复：llmcompressor 保存的 config 是 text-only 架构
  （`Qwen3_5MoeForCausalLM`），但 safetensor 键保留 `model.language_model.*`
  前缀（与 W4A16 完全相同的场景）→ 用
  `scripts/canonicalize_qwen35_text_checkpoint.py` 转换键名，
  生成 `/data/models/test/qtopomoe_nvfp4_text_v1`（123972 键映射，
  shape/dtype/值逐张量验证一致，SHA 见 canonicalization_manifest.json）。
- TP1：health/models/completion/metrics 全通过；c1 1/1、c32 32/32，
  backend = VLLM_CUTLASS；测试后已按 PID 校验并正常停止服务。
- TP2：health/models/completion/metrics 全通过；c32 32/32，
  backend = VLLM_CUTLASS；gate 脚本正常清理服务。
- TP4 质量服务：smoke 32/32；official-like v2 116/116，failed=0、
  truncated=0，服务级 gate accepted。
- 质量独立判定（BF16 必须使用修复后的 `bf16_vllm_tp4_r2`）：
  - C-Eval：49/52 = 94.23%，相对 BF16 0pp；
  - MMLU-Pro：55/64 = 85.94%，相对 BF16 -6.25pp；
  - 合计：104/116 = 89.66%，相对 BF16 **-3.45pp**；
  - 超过预注册 1.5pp 门槛，selfgen v1 的质量 Gate/G1 **rejected**。
- 诊断：所有 correctness flip 都正常 stop、无截断/请求/抽取错误；与已通过的
  RedHatAI checkpoint 覆盖完全相同，30720 个 expert weight global scale
  全相等，差异集中在 activation input scale。256/256 校准 token 序列复算
  完全一致，因此不是数据集或 chat template 差异，而是校准执行状态差异。
- P2P 已生效：8x8 read/write peer matrix 的非对角元素均为 `OK`，服务进程
  未设置 `NCCL_P2P_DISABLE`；P2P 是 BF16/官方/selfgen 的共同控制条件，
  不是本次质量下降原因。

## 4. 后续项（接手人按序执行）

1. **正式 NVFP4 主线**：使用已通过的 RedHatAI checkpoint 采集 route trace
   与真实 M-bucket；selfgen v1 不准入 EP/EPLB。
2. **selfgen 改进分支（可选）**：冻结一个 v2 配方后重新量化。优先消融
   LLM Compressor/compressed-tensors 版本、MoE linearization/independent
   pipeline 和样本执行次序；禁止改评测集、移植 RedHat scale 或覆盖 v1。
   任一候选都必须重跑 coverage、TP1/TP2、c1/c32 和 116 条质量 Gate。
3. **ModelOpt mixed**：Runbook 要求至少两条量化服务线稳定后才开始；当前
   前提已满足，可将 NVIDIA mixed checkpoint（NVFP4 experts + FP8 attention）
   作为独立格式，不可与本 selfgen 纯 NVFP4 结果混写。
4. **项目整体待办**：
   Pareto 6 组正式统计（第 15 步）、release tag/一键 smoke（第 16 步）、
   自研 topology-aware EPLB（Phase 6 待做）仍空缺；若有需要一并推进。

## 5. 环境事实与关键路径

- 服务器：`gpu-111`；仓库：`/home/k8s-ops/moe`；本地镜像仓库：
  `C:\Users\29876\Documents\Codex\2026-08-03\new-chat-2\work\moe-push-20260804`。
- vLLM cleanroom venv：
  `/home/k8s-ops/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129`；
  量化 env：`/data/models/test/qtopomoe_quant_env`
  （llmcompressor 0.12.0.1、compressed-tensors 0.17.1、transformers 5.10.1、
  torch 2.11.0+cu130）。
- BF16 源模型：`/home/k8s-ops/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`。
- 质量 manifest：`/home/k8s-ops/moe/configs/evaluation/official_like_smoke_v2.jsonl`
  （116 条）；BF16 baseline 分数见
  `docs/Q-TopoMoE_Phase2_official_like_v2_BF16_summary_20260805.json`。
- NVFP4 运行目录：`/data/models/test/qtopomoe_quant_runs/nvfp4_*`。
- selfgen 机器可读结论：
  `docs/Q-TopoMoE_Phase2_NVFP4_selfgen_gate_20260809.json`。

## 6. 已遇到问题与解决办法（重要）

1. **GitHub 网络不稳定**：多次 `Failed to connect github.com:443` /
   `Connection was reset`。服务器无法直连 HF（`Network is unreachable`），
   但可连 `hf-mirror.com`、`github.com`、`modelscope.cn`。解决办法：
   推送失败就等网络恢复重试；大文件走 ModelScope。2026-08-09 归档时，
   GitHub 应用的 Git Blob 与 Contents 写接口也返回 403
   `Resource not accessible by integration`，因此不能作为推送替代；不得把
   GitHub `1c79c34` 误报为已同步。
2. **HF 官方 NVFP4 checkpoint 下载失败**：`hf-mirror` 对小文件可用，
   但 safetensors 大文件会 302 重定向到 `us.aws.cdn.hf.co` xet CDN，
   服务器网络不通。解决办法：改用 ModelScope `RedHatAI/...` 镜像下载。
3. **ModelScope 数据集 API 兼容**：旧 `MsDataset.load` 与 datasets 版本
   冲突（`DatasetBuilder.as_dataset() verification_mode` 报错）；
   直接下载 parquet 分片（`/home/k8s-ops/.cache/modelscope/hub/datasets/downloads/`）
   再用 pandas 读取生成 jsonl。
4. **根分区写满（100%）**：`/` 492G 满，导致 audit/量化进程无法创建
   临时文件。已清理 `~/.cache/pip`（11G）、`~/.cache/uv`（8.7G）、
   `~/.cache/vllm/torch_compile_cache`（24G），恢复约 43G（当前 23G 可用）。
   **注意**：后续大任务前先 `df -h /`；编译缓存删除后 vLLM 首次启动
   会重新 torch.compile，启动变慢属正常。
5. **自生成 checkpoint 键名前缀**：llmcompressor 输出 config 为 text-only
   但权重带 `model.language_model.*` 前缀，vLLM 报
   `There is no module or parameter named 'language_model' in Qwen3_5Model`。
   必须用 canonicalize 脚本转换键名后再服务（不要直接 serve 原始输出）。
6. **早前 red1 格误配**（Phase 6 遗留，非 NVFP4）：`ep4_tp2_eplb_red1_p2p`
   当初以与普通 EPLB 相同参数运行（未启用冗余专家），已改名为
   `ep4_tp2_eplb_p2p_run2` 作为重复验证；真正启用冗余专家时 vLLM 报
   `even distribution of experts across ranks`（257 专家为质数，EP2/4/8
   均不整除），已记为框架限制。

## 7. 关键结论（可直接引用）

- 官方 NVFP4 checkpoint 在 SM120/5090 上**真实可加载**（VLLM_CUTLASS
  原生路径），TP1/TP2 + 并发 32 全通过，质量总体达标（-0.86pp），
  满足 Runbook 6.3"先验证官方"前置条件。
- 自生成 NVFP4 量化、canonicalization、TP1/TP2/TP4 服务均跑通，但质量
  总体 -3.45pp，超过门槛，selfgen v1 退出正式系统主线并保留失败证据。
- 已通过的 RedHatAI NVFP4 可进入 route trace/M-bucket；selfgen v1 不准入
  EP/EPLB。后续若做 v2，必须视为新 checkpoint 并完整重跑 Gate。
# 2026-08-12 增量交接

本文件后续的 2026-08-09 快照保留作历史背景；当前权威状态如下：

- 服务器仓库已迁移到 `/data/moe`，模型与实验结果仍在 `/data/models/test`；
- NVFP4 EP8 的在线 placement-plan/迁移/恢复 Gate 已接受，详见[阶段 7–8 正式收尾](results/phase7_phase8_formal_closeout_20260812.md)；
- Phase 8 四候选正式矩阵已完成并接受：108 个 cell、每候选五次重复、共 20 个运行；
- 简化 selector 的 median regret 为 0%，但 p95 regret 为 43.17%，正式 Gate 未通过，禁止在线启用；
- 下一步不是继续在原 108-cell 上调参，而是冻结新特征设计并增加独立测试 workload；full-set FP8/NVFP4 质量评测可与 selector 解耦执行。
