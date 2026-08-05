# Frozen quality evaluation inputs

Generated JSONL and manifest files in this directory are produced by
`evaluation/freeze_quality_sets.py`. They track source revisions, every raw
parquet SHA-256, deterministic sample indices, prompt text, sampling settings,
and BF16/W4A16 prompt-token equivalence.

HumanEval rows are frozen but are not executed on the server until an isolated
code sandbox is available.
