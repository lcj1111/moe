#!/usr/bin/env python3
"""Freeze the full official-protocol evaluation set (benchmark-official setup).

Protocol (aligned with the benchmark authors' own harnesses):
  - MMLU-Pro test (12,032): official 5-shot CoT prompt from
    TIGER-AI-Lab/MMLU-Pro ``evaluate_from_api.py`` (temperature=0, top_p=1,
    penalties=0, max_tokens=4000; answer extracted as "The answer is (X)").
  - C-Eval test (12,342): official answer-only prompt from hkust-nlp/ceval
    (5 dev exemplars per subject, "答案：" answer prefix, A/B/C/D extraction).

``enable_thinking`` is False for both: benchmark-official harnesses do not use
the model's thinking mode; CoT lives in the prompt for MMLU-Pro.

The large JSONL lives outside git; this script writes a small manifest with
SHA-256 and prompt token-ID digest.
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
    MMLU_REVISION,
    mmlu_example,
    mmlu_record,
    sha256_file,
)
from freeze_quality_sets import token_ids


OFFICIAL_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "enable_thinking": False,
}
MMLU_MAX_OUTPUT_TOKENS = 4000
CEVAL_MAX_OUTPUT_TOKENS = 2048


def ceval_official_question(row: dict[str, Any]) -> str:
    """C-Eval answer-only question block: question + 'A. .. D. ..' lines."""
    choices = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    return f"{row['question']}\n{choices}"


def ceval_official_record(
    row: dict[str, Any],
    subject: str,
    index: int,
    demonstrations: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    prompt = (
        f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
    )
    for example in demonstrations:
        prompt += ceval_official_question(example) + f"\n答案：{example['answer']}\n\n"
    prompt += ceval_official_question(row) + "\n答案："
    return {
        "id": f"ceval:{subject}:test:{row.get('id', index)}",
        "benchmark": "ceval",
        "protocol": "official_answer_only_full",
        "source_index": index,
        "subject": subject,
        "messages": [{"role": "user", "content": prompt}],
        "answer": str(row["answer"]).strip().upper(),
        "score_type": "multiple_choice",
        "max_tokens": CEVAL_MAX_OUTPUT_TOKENS,
        "sampling": {**OFFICIAL_SAMPLING, "seed": seed},
    }


def mmlu_official_record(
    row: dict[str, Any],
    index: int,
    demonstrations: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    record = mmlu_record(row, index, demonstrations, seed)
    record["protocol"] = "official_cot_fewshot_full"
    record["max_tokens"] = MMLU_MAX_OUTPUT_TOKENS
    record["sampling"] = {**OFFICIAL_SAMPLING, "seed": seed}
    return record


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
        rows.append(mmlu_official_record(
            row, index, mmlu_demos[str(row["category"])], args.seed
        ))
    for path in ceval_test_paths:
        subject = path.stem
        subject_rows = pq.read_table(path).to_pylist()
        for index, row in enumerate(subject_rows):
            rows.append(ceval_official_record(
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
        "schema_version": "qtopomoe.official_full.v1",
        "path": str(args.output.resolve()),
        "records": len(rows),
        "counts": counts,
        "seed": args.seed,
        "revisions": {
            "mmlu_pro": MMLU_REVISION,
            "ceval_parquet": CEVAL_PARQUET_REVISION,
        },
        "fewshot": {"mmlu_pro": 5, "ceval": 5},
        "max_output_tokens": {
            "mmlu_pro": MMLU_MAX_OUTPUT_TOKENS,
            "ceval": CEVAL_MAX_OUTPUT_TOKENS,
        },
        "sampling": OFFICIAL_SAMPLING,
        "sha256": sha256_file(args.output),
        "prompt_token_ids_sha256": token_digest.hexdigest(),
        "prompt_tokens": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
        },
        "token_ids_equal": True,
        "note": (
            "Benchmark-official protocol: MMLU-Pro 5-shot CoT (temperature 0, "
            "max_tokens 4000) and C-Eval 5-shot answer-only; enable_thinking "
            "is False for both. Large JSONL is kept outside git; this manifest "
            "pins hashes and protocol."
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
