# 冻结的质量评测输入

本目录中的 JSONL 与 manifest 由 `evaluation/freeze_quality_sets.py` 生成，记录源数据 revision、每条原始 parquet 的 SHA-256、确定性的样本索引、提示词、采样参数，以及 BF16/W4A16 的 prompt-token 等价性。

HumanEval 行已冻结，但在没有隔离代码沙箱之前，不会发送到推理服务执行。

`quality_smoke_v1.jsonl` 与 `quality_formal_v1.jsonl` 是确定性的 zero-shot、关闭 thinking 的回归输入。它们的绝对分数不能与 Qwen 发布的 benchmark 分数直接比较，只用于在相同输入下检测 BF16、量化模型和运行时之间的质量变化。

官方协议风格的采样集由 `evaluation/freeze_official_like_smoke.py` 单独生成，使用固定的 MMLU-Pro validation CoT 示例、每个 C-Eval 科目 5 个 dev 示例、thinking 模式和 Qwen 推荐采样参数。这仍然是 sampled reproduction；只有使用明确冻结的官方 harness 完成 full-set，才能称为官方分数复现。

## Full-set 官方协议资产

`full_set_official_v1.manifest.json` 固定了 full-set 协议：MMLU-Pro test（12,032 条）加 C-Eval test（12,342 条），共 24,374 条记录；seed=42、5+5 few-shot、thinking 模式、输出上限 32,768，并记录 93 MB JSONL（`full_set_official_v1.jsonl`，不入 Git）的 SHA-256。生成流程为：

1. `evaluation/fetch_official_protocol_assets.py`
2. `evaluation/freeze_full_set.py`
3. `evaluation/slice_pilot.py` 先切出小规模计时 pilot，再决定是否执行多小时 full-set。
