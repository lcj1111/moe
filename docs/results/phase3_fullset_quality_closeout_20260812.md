# Phase 3 全量质量验收

## 结论

在统一评测协议和 24,330 条三格式共同样本上，BF16、FP8 与 NVFP4 的准确率分别为
86.9955%、86.9749% 和 86.3009%。FP8 与 BF16 基本持平；NVFP4 相对 BF16 下降
0.6946 个百分点，满足项目设定的不超过 1.5 个百分点质量门槛。

| 数据集 | 共同样本 | BF16 | FP8 | NVFP4 | FP8−BF16 | NVFP4−BF16 |
|---|---:|---:|---:|---:|---:|---:|
| 总体 | 24,330 | 86.9955% | 86.9749% | 86.3009% | -0.0206 pp | -0.6946 pp |
| C-Eval | 12,306 | 89.0704% | 89.3223% | 88.4853% | +0.2519 pp | -0.5851 pp |
| MMLU-Pro | 12,024 | 84.8719% | 84.5725% | 84.0652% | -0.2994 pp | -0.8067 pp |

总体正确数为 BF16 21,166、FP8 21,161、NVFP4 20,997。BF16 与 FP8 仅相差
5 题，分科增减方向相反，因此本项目只将其表述为“冻结协议下总体近似持平”。

## 评测口径

- 数据：C-Eval 与 MMLU-Pro，共 24,374 条冻结样本。
- 比较集合：仅使用 BF16、FP8、NVFP4 三者均可评分的共同 ID。
- 一致性：模型以外的 tokenizer、chat template、seed、采样参数和答案提取规则保持一致。
- 审计：比较前检查 ID 唯一性、身份字段和输入完整性。
- 统计单位：表中差值均为绝对百分点（pp），不是相对百分比。

共同样本的正确性组合如下：

| BF16 / FP8 / NVFP4 | 样本数 |
|---|---:|
| 三者都正确 | 20,048 |
| 仅 BF16、FP8 正确 | 534 |
| 仅 BF16、NVFP4 正确 | 311 |
| 仅 BF16 正确 | 273 |
| 仅 FP8、NVFP4 正确 | 317 |
| 仅 FP8 正确 | 262 |
| 仅 NVFP4 正确 | 321 |
| 三者都错误 | 2,264 |

## 复现与证据

共同分母比较由
[`scripts/compare_fullset_quality_results.py`](../../scripts/compare_fullset_quality_results.py)
执行。机器可读结果见
[`Q-TopoMoE_Phase3_BF16_FP8_NVFP4_质量总结_20260816.json`](../Q-TopoMoE_Phase3_BF16_FP8_NVFP4_质量总结_20260816.json)。

- 比较结果目录：`/data/models/test/qtopomoe_quality_runs/full_official_three_format_compare_v1`
- 比较 JSON SHA-256：`2e34750125da1f420b07952e144b88cd93ab906919ab894422a993b6a2e753c9`
- BF16 输入 SHA-256：`98b10220241353f28e8ab31a3d2a17178ef8a613a565a88edc49bdb14863b1f2`
- FP8 输入 SHA-256：`6dc2df8a92f83eabd657d43cf1e77aa12a7c2088a40ed3061c50d45c3eb4851f`
- NVFP4 输入 SHA-256：`d8e17e7106b2575ff1ac7b020ddc7d93f7d3d5698794310b163b959d75601920`

结果只适用于仓库冻结的评测协议、checkpoint、tokenizer/chat template、seed 与采样参数，
不能与使用不同评测框架的模型卡分数直接比较。
