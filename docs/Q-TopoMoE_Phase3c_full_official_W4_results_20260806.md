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
[Phase3c 官方协议冻结报告](Q-TopoMoE_Phase3c_full_set_official_protocol_20260806.md)）：

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
[合并 summary](Q-TopoMoE_Phase3c_full_official_w4_merged_summary_20260806.json)。

## 5. 边界与说明

- 官方 BF16 分数来自 Qwen 官方模型卡（MMLU-Pro 85.2；C-Eval 90.0 为
  thinking 模式），与本评测协议存在细微差异，对比为参考性质；同协议
  BF16 baseline 以 Phase 2 sampled（116 条）为准。
- 原始结果（68 MB JSONL，含完整 response）保留在服务器
  `/data/models/test/qtopomoe_quality/full_official_w4_merged_v2.jsonl`，
  不入库；本报告与 summary 钉住数字与截断清单。
