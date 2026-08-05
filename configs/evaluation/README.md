# Frozen quality evaluation inputs

Generated JSONL and manifest files in this directory are produced by
`evaluation/freeze_quality_sets.py`. They track source revisions, every raw
parquet SHA-256, deterministic sample indices, prompt text, sampling settings,
and BF16/W4A16 prompt-token equivalence.

HumanEval rows are frozen but are not executed on the server until an isolated
code sandbox is available.

`quality_smoke_v1.jsonl` and `quality_formal_v1.jsonl` are deterministic,
zero-shot, non-thinking regression inputs. Their absolute scores must not be
compared with Qwen-published benchmark numbers. They are used to detect quality
changes between BF16 and quantized/runtime candidates under identical inputs.

The official-like sampled protocol is generated separately by
`evaluation/freeze_official_like_smoke.py`. It uses the pinned MMLU-Pro
validation CoT demonstrations, five C-Eval dev examples per subject, thinking
mode, and Qwen-recommended sampling. It remains a sampled reproduction; only a
full-set run with an explicitly frozen official harness may be called an
official-score reproduction.
