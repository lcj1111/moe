# Q-TopoMoE Phase 3：全量 route trace 采集与漂移分析报告

> 生成日期：2026-08-06（Asia/Shanghai）
> 数据范围：official-like v2 冻结样本 116 条（MMLU-Pro 64 + C-Eval 52）
> 原始 trace（npy 分片、traces.jsonl、histogram）体积大，不入库；采集
> manifest 已归档到 `traces/manifests/`，钉住模型路径、prompt 哈希、shape
> 与文件 SHA-256。

## 1. 采集配置

两套全量采集使用完全相同的 prompt 清单（`configs/evaluation/official_like_smoke_v2.jsonl`）、
seed=42、gen_tokens=16、TP4、GPU0-3，仅模型与 MoE backend 不同：

| 项目 | BF16 | W4A16 |
|---|---|---|
| Model | Qwen3.6-35B-A3B BF16 snapshot | canonical W4A16 text v1 |
| MoE backend | auto | triton |
| gen_tokens | 16 | 16 |
| prompt 覆盖 | 116/116 | 116/116 |
| traces_rows | 4,793,080 | 4,793,080 |
| token 总数（npy 拼接） | 119,827 | 119,827 |

采集机制为 vLLM cleanroom（commit `33c50587d`）原生
`enable_return_routed_experts`，输出 `routed_experts` 数组 shape
`[seq_len, 40, top-8]`（uint8）。**只捕获 expert IDs；`topk_weights` 与
`expert_service_time_us` 为 null**（vLLM 该机制不导出），因此本报告不含
router-probability KL，也不伪造 kernel 延迟数据。

采集脚本：`traces/capture_routes.py`；采集后自动生成
`capture_manifest.json`（含每个 npy 的 SHA-256）与
`expert_token_histogram.json`（M-bucket 分布）。

## 2. 漂移指标（BF16 reference → W4A16-triton candidate）

分析脚本：`analysis/route_drift.py`；全量输出见
[Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json](Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json)。

| 指标 | 全量值 | 12 题 pilot | 说明 |
|---|---:|---:|---|
| mean_jaccard | 0.824 | 0.825 | token 级 top-8 平均 Jaccard |
| mean_flip_rate | 59.6% | 59.5% | 同 token 分配集合完全不同的比例 |
| mean_count_correlation | 0.9995 | 0.999 | 每层专家计数相关 |
| ref_count_cv / cand_count_cv | 1.008 / 1.017 | — | 计数负载离散系数 |
| M-bucket delta | 各桶 ±2 以内 | — | histogram 分布差异极小 |

全量结果与 12 题 pilot 高度一致，说明 pilot 采样有代表性。

## 3. 分层趋势

漂移随层深单调增大，深层最不稳定：

| 层段 | mean_jaccard | mean_flip_rate |
|---|---:|---:|
| L0-9（浅） | 0.914 | 35.3% |
| L10-29（中） | 0.811 | 64.9% |
| L30-39（深） | 0.761 | 73.2% |

最差五层集中在 L35-39（jaccard 0.740-0.756，flip 74-77%）；L0 最稳
（jaccard 0.986，flip 5.3%）。

## 4. 结论

1. **分配层面存在明显 token 级噪声**：平均约 6 成 token 的 top-8 集合在
   W4A16（triton）下与 BF16 不同，且集中在深层。
2. **聚合层面高度一致**：专家计数相关 0.9995、M-bucket delta 各桶 ±2 以内、
   CV 几乎不变，说明量化没有改变“每个专家接收多少 token”的宏观负载形态。
3. 对 Phase 4/8 的影响：以 expert 计数/M-bucket 驱动的 kernel 调度与 EPLB
   决策，可基于 BF16 trace 设计、用 W4 验证；若后续依赖单个 token 的专家
   归属（如 top-1 路由），需保留 6 成 flip 的余量。
4. **局限**：无 router 概率（KL 不可算），无服务延迟 trace，不能推导
   kernel 级 latency；这两项分别依赖 vLLM 导出扩展与 Phase 4 实测。

## 5. 下一步

- Full-set 官方协议评测资产已冻结（24,374 条，见
  `configs/evaluation/full_set_official_v1.manifest.json`），先跑 32 条计时
  pilot（`evaluation/slice_pilot.py` + `clients/quality_eval.py`）推算全量耗时。
- 全量漂移分析结论写入 Phase 8 候选筛选输入。
