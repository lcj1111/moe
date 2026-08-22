# 数据 manifest 说明

`data/manifests/` 只保存输入数据的版本、样本数量和完整性信息，不保存大体积原始数据本身。

当前机器本地的 smoke seed 由 `scripts/freeze_baseline.sh` 复制到 `sharegpt_seed.json`，并通过 SHA-256 校验。该文件包含 64 条记录，只用于服务 smoke 和筛选实验。

正式的 256 条校准集以及 benchmark 评测 manifest 不会从 64 条 smoke seed 人工拼造。它们必须来自固定的数据集 revision，并在量化或质量结论前记录：数据集 revision、样本索引、tokenizer revision、chat template、seed 和 SHA-256。
