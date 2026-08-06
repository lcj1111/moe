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
