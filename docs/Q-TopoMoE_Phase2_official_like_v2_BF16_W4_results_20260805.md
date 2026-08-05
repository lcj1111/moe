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
