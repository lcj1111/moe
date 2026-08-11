# 阶段 3：route trace 与 official-like 评测

> 以下为按日期合并的历史报告。源内容已保留，仅规范化了行尾空格；SHA-256 按 UTF-8 Git blob（LF 换行）计算，机器可读产物保持原始路径以确保复现。

## 源文件完整性

| 原始文件 | UTF-8 字节数 | Git blob 的 SHA-256 |
|---|---:|---|
| `docs/Q-TopoMoE_Phase3_route_trace_full_drift_20260806.md` | 3687 | `A8A5394F3CB33B8316776F5F06AC55D6354AEEEF1BF5EFF011E6EEB8ADC2D268` |
| `docs/Q-TopoMoE_Phase3c_full_set_official_protocol_20260806.md` | 5126 | `BCDDFFB2A6640878E88BC5930A3BE05701D8AECFD7F6910AF3DE81EC9A3F1EFC` |
| `docs/Q-TopoMoE_Phase3c_full_official_W4_results_20260806.md` | 3043 | `E6206F2E4B03FC0F0A116A7FFA066F6A50C5F6CFFC0591C08E01106C2D6A673F` |
| `docs/Q-TopoMoE_Phase3_NVFP4_route_and_Phase7_EPLB_20260809.md` | 5414 | `0E3BBA6B2AE986CEF97FAA1B43FF21BC10FFF8A173587F1EC239403ADB1D2E65` |

---

## 源文件： `docs/Q-TopoMoE_Phase3_route_trace_full_drift_20260806.md`

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
[Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json](../Q-TopoMoE_Phase3_route_drift_bf16_vs_w4_20260806.json)。

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

---

## 源文件： `docs/Q-TopoMoE_Phase3c_full_set_official_protocol_20260806.md`

# Q-TopoMoE Phase 3c：full-set 官方协议冻结与计时 pilot

> 生成日期：2026-08-06（Asia/Shanghai）
> 结论先行：MMLU-Pro 与 C-Eval 官方 harness 均**不要求 thinking 模式**；
> 切换为 benchmark-official 协议后单题耗时约降为原来的 1/4，全量 24,374
> 条评测从“数十天”压缩到可按 8 卡数天内完成。

## 1. 官方协议核实（来源：benchmark 作者官方仓库）

### MMLU-Pro（TIGER-AI-Lab/MMLU-Pro，NeurIPS 2024）

- 官方脚本 `evaluate_from_api.py`：5-shot，每 category 取 validation 集的
  CoT 示例（`cot_content`）。
- Prompt 模板："The following are multiple choice questions (with answers)
  about {category}. Think step by step and then output the answer in the
  format of \"The answer is (X)\" at the end."
- 采样：`temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0,
  max_tokens=4000`。
- 答案提取：正则 `answer is \(?([A-J])\)?`，失败降级到末尾字母。
- **无 thinking 开关**：CoT 在 prompt 内，不在模型推理模式。

### C-Eval（hkust-nlp/ceval，NeurIPS 2023）

- 每 subject 5 个 dev 示例（few-shot，含 explanation）；也支持 zero-shot。
- 官方提供 answer-only 与 chain-of-thought 两种 prompt；答案提取用正则
  取 A/B/C/D，或 constrained decoding（对 A/B/C/D 算概率）。
- **无 thinking 开关**；官方注明 constrained decoding 不适用于 CoT。
- 2025-07-27 官方发布完整 test split，本地可直接评测。

### 与旧协议的区别

上一版冻结（`full_set_official_v1.jsonl`）采用 Qwen3 chat 默认采样
（`enable_thinking=true, temperature=1.0, top_p=0.95, top_k=20,
presence_penalty=1.5`），不是 benchmark 官方协议。新冻结
（`full_set_official_protocol_v1.jsonl`）逐项对齐官方 harness。

## 2. 官方协议冻结结果

冻结脚本：`evaluation/freeze_full_set_official.py`（新）；
manifest：`configs/evaluation/full_set_official_protocol_v1.manifest.json`。

| 项 | 值 |
|---|---|
| 总条数 | 24,374（MMLU-Pro test 12,032 + C-Eval test 12,342） |
| MMLU-Pro | 5-shot CoT，temp 0，max_tokens 4000，thinking 关 |
| C-Eval | 5-shot answer-only（"答案："前缀），temp 0，max_tokens 2048，thinking 关 |
| JSONL SHA-256 | `73fa596dff4ee71055db6e04a71c5dbca640c31c18e1930ca1063466b4c1d1f8` |
| prompt tokens | min 254 / max 2897 / mean 941（BF16 与 W4 tokenizer 完全一致） |

92 MB JSONL 不入库，manifest 钉住全部哈希与协议参数。

## 3. 计时 pilot 对比（BF16，vLLM cleanroom，TP4 GPU0-3，concurrency 1）

### thinking 模式 pilot（旧协议，32 条 = MMLU 16 + C-Eval 16）

| 项 | MMLU-Pro | C-Eval |
|---|---:|---:|
| 单题 request_ms | 中位 86.9s / 平均 96.0s | 中位 122.6s / 平均 137.0s |
| 输出 tokens | 平均 1,421 | 平均 2,052 |
| 完成 | 16/16 | 16/16 |

32 条串行约 1.04 小时。注：该 pilot 为连续切片，MMLU 全为 business、
C-Eval 全为 accountant，分布有偏，仅作对比参考。

### 官方协议 pilot（新协议，8 条 = 分层抽样 MMLU 4 + C-Eval 4）

| 项 | MMLU-Pro | C-Eval |
|---|---:|---:|
| 单题 request_ms | 中位 30.9s / 平均 33.4s | 中位 13.4s / 平均 27.1s |
| 输出 tokens | 平均 512 | 平均 415 |
| 完成 | 4/4（无截断/错误） | 4/4（无截断/错误） |

8 条串行约 4.0 分钟。分层抽样覆盖 MMLU 4 个 category（other/business/
biology/economics）与 C-Eval 4 个科目（middle_school_politics/
clinical_medicine/advanced_mathematics/sports_science）。

### 对比结论

官方协议单题平均耗时约 30.2s（中位 23.6s），约为 thinking 模式
（平均 116s）的 **1/4**；输出 token 从平均 1,737 降至 463。加速主要来自
关闭 thinking + 更短的 max_tokens，而非吞吐变化。

## 4. 全量 24,374 条时间外推

按官方协议 pilot 分 benchmark 均值外推（concurrency=1 串行下界）：

| 组合 | 预估耗时 |
|---|---:|
| MMLU-Pro 12,032 × 33.4s | 111.7 h |
| C-Eval 12,342 × 27.1s | 92.9 h |
| **串行合计** | **204.6 h（约 8.5 天）** |
| 8 卡 = 2×TP4 服务 × concurrency 4（按 3x 有效加速估算） | 约 34 h（1.5 天） |
| 8 卡 = 2×TP4 服务 × concurrency 8（按 5x 有效加速估算） | 约 21 h（<1 天） |

注：并发有效加速系数为估算，实际以短并发实测校准；单题时间受
advanced_mathematics 类长题（pilot 中 69.4s）拖尾影响。

## 5. 执行方案（8 卡）

用户已授权全量评测时终止 GPU4-7 上的 llama-server（OpenMOSE-262B IQ1，
port 8080，PID 见 gate 记录），释放全部 8 卡：

- **2×TP4 双服务**：GPU0-3 与 GPU4-7 各起一个 vLLM BF16 TP4 服务
  （cleanroom `33c50587d`），共享同一官方协议 manifest；
- **分片**：按行号 1/2 切分，两个 `quality_eval --concurrency N` 并行，
  输出合并后统一评分；
- **校验**：每片 gate accepted + 无截断 + 无错误；合并后按 benchmark 出
  准确率，与 Phase 2 sampled 结果可比。

结果与计时摘要归档见同目录
`Q-TopoMoE_Phase3c_pilot_*_20260806.json`。

---

## 源文件： `docs/Q-TopoMoE_Phase3c_full_official_W4_results_20260806.md`

# Q-TopoMoE Phase 3c：W4A16 full-set 官方协议评测结果

> 生成日期：2026-08-06（Asia/Shanghai）
> 模型：Qwen3.6-35B-A3B canonical W4A16（`/data/models/test/qtopomoe_w4a16_canonical_text_v1`）
> 后端：vLLM cleanroom `33c50587d`，MoE backend=triton，8×RTX 5090（2×TP4 双服务）

## 1. 结论

W4A16（Triton MoE）在 full-set 官方协议评测上与官方 BF16 分数基本持平：

| Benchmark | 记录 | 已计分 | 截断 | 正确 | 准确率 | 官方 BF16 参考 |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro（test） | 12,032 | 12,032 | 0 | 10,177 | **84.58%** | 85.2（模型卡） |
| C-Eval（test） | 12,342 | 12,321 | 21 | 10,999 | **89.27%** | 90.0（thinking） |

MMLU-Pro 与官方 BF16 差 0.62 点，C-Eval 差 0.73 点，与 Phase 2 的 116 条
sampled 对比结论（drop≤5 gate pass）一致：**W4A16 量化未造成明显质量损失**。

## 2. 评测协议

与 benchmark 官方 harness 对齐（详见
[Phase3c 官方协议冻结报告（本合并文件）](phase3_route_and_official_eval.md)）：

- MMLU-Pro：5-shot CoT，temperature 0，max_tokens 4000，thinking 关；
  答案按 "The answer is (X)" 正则提取。
- C-Eval：5-shot answer-only（"答案："前缀），temperature 0，max_tokens 2048，
  thinking 关；答案按 A/B/C/D 正则提取。

冻结输入：`configs/evaluation/full_set_official_protocol_v1.manifest.json`
（24,374 条，SHA-256 钉住）。大 JSONL 不入库。

## 3. 执行过程

三轮运行（8 卡 2×TP4 → 4 卡 TP4）：

| 轮次 | 范围 | max_tokens | 结果 |
|---|---|---|---|
| 主跑 | 24,374 条（A/B 分片） | MMLU 4000 / C-Eval 2048 | 1,206 条截断 |
| 补跑 1 | 1,206 条 | MMLU 8000 / C-Eval 4096 | 1,004 条成功，剩 202 条 |
| 补跑 2 | 202 条 | MMLU 16384 / C-Eval 8192 | MMLU 全部完成；C-Eval 剩 21 条 |

提速说明：gate 脚本去掉 `--enforce-eager` 后启用 CUDAGraph/torch.compile，
生成吞吐由 ~59 tok/s 提升至 ~700 tok/s（约 12 倍），全量评测从数十小时
压缩至约 4-5 小时。

## 4. 剩余截断（21 条 C-Eval）

21 条在 8192 token 内仍未输出答案，集中在数学/长计算类科目
（civil_servant 4、accountant 2、discrete_mathematics 2、
advanced_mathematics 2 等）。响应尾部显示模型进入"最终猜测/自我怀疑"
状态，继续提高上限预期难以收敛。按官方 answer-only 协议视为未完成、
不计分（占总量 0.17%），完整 ID 清单见
[合并 summary](../Q-TopoMoE_Phase3c_full_official_w4_merged_summary_20260806.json)。

## 5. 边界与说明

- 官方 BF16 分数来自 Qwen 官方模型卡（MMLU-Pro 85.2；C-Eval 90.0 为
  thinking 模式），与本评测协议存在细微差异，对比为参考性质；同协议
  BF16 baseline 以 Phase 2 sampled（116 条）为准。
- 原始结果（68 MB JSONL，含完整 response）保留在服务器
  `/data/models/test/qtopomoe_quality/full_official_w4_merged_v2.jsonl`，
  不入库；本报告与 summary 钉住数字与截断清单。

---

## 源文件： `docs/Q-TopoMoE_Phase3_NVFP4_route_and_Phase7_EPLB_20260809.md`

# 阶段 3 NVFP4 路由 trace 与阶段 7 EPLB 检查点

## 已接受的 route-trace Gate

RedHatAI NVFP4 检查点使用与 BF16/W4A16 相同的冻结协议采集：116 条 prompt、manifest 顺序 0..115、seed 42、生成 16 token、GPU0-3 TP4、`max_model_len=8192`。运行时证据确认使用原生 `VLLM_CUTLASS` MoE backend。

- prompt 数量：116/116；
- prompt ID/顺序、prompt-token SHA-256 与 token 数：全部匹配 BF16；
- token 位置：119,827；
- layer-token 行：4,793,080（`[token, 40, 8]`、`uint8`）；
- expert 范围：0..255；
- 116 个数组 SHA-256：全部匹配 capture manifest；
- 没有出现 `routed_experts is None`；
- capture manifest SHA-256：`114c2180d15d85b94a96b4c6914c24496f4422bb0a400e9e21d7d5646efdd4ab`。

BF16 到 NVFP4 的路由漂移为 `mean_jaccard=0.794115`、`mean_flip_rate=0.660875`、`mean_count_correlation=0.99898`。也就是说，量化改变了许多 token 级 top-k 成员，但整体 expert 负载形状仍高度接近。该结果是 placement 的输入，不替代已经接受的质量 Gate。

trace 得到的实际 workload 混合比例为 M=1：0.007198，M=2048：0.897493，M=8192：0.095309。

## 量化感知的 expert 字节数

在不加载模型张量的情况下审计 Safetensors metadata。40×256 个 routed expert 全部存在；每个 expert 包含 12 个存储张量，精确大小为 1,769,496 bytes，routed-expert 总存储为 18,119,639,040 bytes。审计使用 packed U8 权重、FP8 scale 和 FP32 global scale 的真实存储，而不是按名义 4-bit 权重估算。

## 拓扑与迁移基础操作

`configs/experiments/topology_gpu111.yaml` 已记录当前 P2P 状态：所有非对角读写对均为 `OK`。正式 NCCL 五次运行数据仍是 mapping 来源；保留 NODE 优先分组，是因为主机实测 NCCL 结果优于 PIX 负控制。

精确大小（1,769,496-byte）的迁移 microbench，重复五次：

| primitive | mean us | p95 us | effective GB/s |
|---|---:|---:|---:|
| logical same-GPU remap | 0.0421 | 0.0424 | n/a |
| same-GPU copy | 7.8008 | 7.8525 | 226.84 |
| same-NUMA PIX 0->1 | 42.6011 | 42.7145 | 41.54 |
| same-NUMA NODE 0->2 | 43.1823 | 43.4161 | 40.98 |
| cross-NUMA SYS 0->4 | 66.1044 | 66.6786 | 26.77 |

所有复制均验证内容一致，并报告 peer access enabled。这些只是基础操作成本；在线迁移接受前仍必须测量 block time、恢复时间和受影响服务的 p99。

## 原生 NVFP4 kernel 与 EP 准入

原生 vLLM CUTLASS NVFP4 MoE 基础操作按真实 trace bucket 重复 50 次：M=1、2048、8192 的 p95 分别为 250.42、728.23、2219.58 us。使用冻结的 token 加权 M 混合并除以 `M * top_k`，得到可审计的离线 placement proxy：每个 routed assignment 为 0.268434768470764 us；这不是端到端服务延迟。

官方 RedHatAI NVFP4 通过两个真实 expert-parallel 准入 cell：

| cell | actual EP ranks | backend | completed/failed | e2e p50/p95 ms |
|---|---:|---|---:|---:|
| TP4 + EP4 static | 4 | MARLIN | 32/0 | 751.95 / 1682.29 |
| TP8 + EP8 static | 8 | MARLIN | 32/0 | 823.76 / 1996.98 |

非 EP 服务使用 `VLLM_CUTLASS`，分片 EP cell 使用 `MARLIN`，这是运行时选择且已明确记录；CUTLASS microbench 行不冒充 EP MARLIN 计时。EP8 峰值 HBM 为每 GPU 29027--29109 MiB；扣除每卡精确 1280 个本地 routed expert 后，实测非 expert/KV headroom 为 26.24--26.32 GiB。

## EPLB 实现状态

`selector/eplb_policy.py` 现在提供：

- 按 expert 负载、精确量化字节数、HBM headroom、源 GPU 权重和实测 pair cost 做确定性离线 placement；
- 可选的冗余 expert placement；
- 预测跨 NUMA 字节数、dispatch cost、HBM 使用、迁移字节数和稳定的 plan SHA-256；
- Runbook 在线状态机（500 ms/1000 requests、EMA 0.2、连续三个窗口 CV >0.25、benefit >=5%、benefit/cost >=2、residency 10、cooldown 20，连续三个 p99 回退超过 5% 窗口后回滚）。

按实测输入生成的全域 plan 覆盖 10,240 个 expert；精确 expert bytes、已接受 route load、实测 kernel proxy、实测 EP8 HBM headroom 与实测 topology 均已绑定。八张 GPU 的 load span 小于 0.81 us，稳定 plan SHA-256 为 `d53bb6653abed0fe67163888dd838a8f2aef60d2d86968cc89f0c3d0430865d6`。在完成在线服务迁移的 block/recovery/p99 影响及 placement-plan application 验证前，整体 `formal_ready` 仍为 false。

原生 vLLM EPLB admission cell 也在真实 TP8/EP8 world 上执行，但在模型构造阶段、尚未服务前失败：`NotImplementedError: EPLB is not supported CompressedTensorsW4A4Nvfp4MoEMethod.` 冻结的 vLLM 构建与当前 upstream main 对该 compressed-tensors NVFP4 method 都保持 EPLB disabled。历史 upstream 实现支持的是另一条 `ModelOptNvFp4FusedMoE` 路径，不能证明修改当前路径的 capability property 是安全的。因此项目不对 serving venv 做 hot-patch；静态 EP4/EP8 继续允许，原生 EPLB 与自定义运行时 plan 应用标记为 unavailable/pending，而不是报告为成功。