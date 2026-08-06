#!/usr/bin/env python3
"""Freeze the full official-like evaluation set (protocol-aligned, full test
splits): MMLU-Pro test (12,032) + C-Eval test (13,948), 5-shot CoT prompts.

The manifest is large (tens of MB) and is expected to live outside git;
this script records the JSONL SHA-256 and per-prompt token-ID digest in a
small ``.manifest.json`` that is safe to commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import defaultdict
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from freeze_official_like_smoke import (
    CEVAL_PARQUET_REVISION,
    LETTERS,
    MAX_OUTPUT_TOKENS,
    MMLU_REVISION,
    SAMPLING,
    ceval_question,
    mmlu_example,
    mmlu_record,
    sha256_file,
)
from freeze_quality_sets import token_ids


def ceval_test_record(
    row: dict[str, Any],
    subject: str,
    index: int,
    demonstrations: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    prompt = (
        f"以下是关于 {subject} 的单项选择题（含答案示例）。请逐步思考，"
        '最后仅输出 JSON，例如 {"answer": "C"}。\n\n'
    )
    for example in demonstrations:
        prompt += ceval_question(example) + f"\n答案：{example['answer']}\n\n"
    prompt += ceval_question(row)
    return {
        "id": f"ceval:{subject}:test:{row.get('id', index)}",
        "benchmark": "ceval", "protocol": "official_like_5shot_full",
        "source_index": index, "subject": subject,
        "messages": [{"role": "user", "content": prompt}],
        "answer": str(row["answer"]).strip().upper(),
        "score_type": "multiple_choice", "max_tokens": MAX_OUTPUT_TOKENS,
        "sampling": {**SAMPLING, "seed": seed},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-a", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-b", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = args.raw_root.resolve()

    mmlu_test = pq.read_table(raw / "mmlupro_test.parquet").to_pylist()
    mmlu_dev = pq.read_table(raw / "mmlupro_validation.parquet").to_pylist()
    mmlu_demos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mmlu_dev:
        mmlu_demos[str(row["category"])].append(row)
    if sorted(len(rows) for rows in mmlu_demos.values()) != [5] * 14:
        raise RuntimeError("MMLU-Pro validation must contain 5 CoT rows per category")

    ceval_test_paths = sorted((raw / "ceval_test").glob("*.parquet"))
    ceval_dev_paths = sorted((raw / "ceval_dev").glob("*.parquet"))
    if len(ceval_test_paths) != 52 or len(ceval_dev_paths) != 52:
        raise RuntimeError("C-Eval requires 52 test and 52 dev subject files")
    dev_by_subject = {
        path.stem: pq.read_table(path).to_pylist() for path in ceval_dev_paths
    }

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(mmlu_test):
        rows.append(mmlu_record(
            row, index, mmlu_demos[str(row["category"])], args.seed
        ))
    for path in ceval_test_paths:
        subject = path.stem
        subject_rows = pq.read_table(path).to_pylist()
        for index, row in enumerate(subject_rows):
            rows.append(ceval_test_record(
                row, subject, index, dev_by_subject[subject], args.seed
            ))

    tokenizer_a = AutoTokenizer.from_pretrained(args.tokenizer_a, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(args.tokenizer_b, trust_remote_code=True)
    token_digest = hashlib.sha256()
    token_lengths: list[int] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            ids_a = token_ids(tokenizer_a, row["messages"])
            ids_b = token_ids(tokenizer_b, row["messages"])
            if ids_a != ids_b:
                raise RuntimeError(f"tokenizer mismatch for {row['id']}")
            token_lengths.append(len(ids_a))
            token_digest.update(json.dumps(ids_a, separators=(",", ":")).encode())
            token_digest.update(b"\n")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts = {
        "mmlu_pro": sum(1 for r in rows if r["benchmark"] == "mmlu_pro"),
        "ceval": sum(1 for r in rows if r["benchmark"] == "ceval"),
    }
    manifest = {
        "schema_version": "qtopomoe.official_like_full.v1",
        "path": str(args.output.resolve()), "records": len(rows),
        "counts": counts, "seed": args.seed,
        "revisions": {
            "mmlu_pro": MMLU_REVISION,
            "ceval_parquet": CEVAL_PARQUET_REVISION,
        },
        "fewshot": {"mmlu_pro": 5, "ceval": 5},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "sampling": SAMPLING, "sha256": sha256_file(args.output),
        "prompt_token_ids_sha256": token_digest.hexdigest(),
        "prompt_tokens": {
            "min": min(token_lengths), "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
        },
        "token_ids_equal": True,
        "note": (
            "Full official-like test splits with the model-card recommended "
            "32,768 output ceiling. Large JSONL is kept outside git; this "
            "manifest pins hashes and protocol."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
