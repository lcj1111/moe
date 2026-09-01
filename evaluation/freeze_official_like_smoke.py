#!/usr/bin/env python3
# 作用：从官方协议数据中冻结分层抽样的轻量 smoke 集。
"""Freeze a sampled, official-like MMLU-Pro and C-Eval protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import defaultdict
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from freeze_quality_sets import sample_indices, stratified_indices, token_ids

LETTERS = "ABCDEFGHIJ"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
CEVAL_MAIN_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"
CEVAL_PARQUET_REVISION = "8267189d6ba0d516d414a98919558032958c4466"
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "enable_thinking": True,
}
MAX_OUTPUT_TOKENS = 32768


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mmlu_example(row: dict[str, Any], cot: str) -> str:
    options = "".join(
        f"{LETTERS[index]}. {option}\n"
        for index, option in enumerate(row["options"])
        if option != "N/A"
    )
    if cot.startswith("A: "):
        cot = cot[3:]
    return f"Question: {row['question']}\nOptions: {options}Answer: {cot}\n\n"


def mmlu_record(
    row: dict[str, Any], index: int, demonstrations: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{row['category']}. Think step by step and then output the answer in "
        'the format of "The answer is (X)" at the end.\n\n'
    )
    for example in demonstrations:
        prompt += mmlu_example(example, str(example["cot_content"]))
    prompt += mmlu_example(row, "Let's think step by step.")
    return {
        "id": f"mmlu_pro:test:{row.get('question_id', index)}",
        "benchmark": "mmlu_pro", "protocol": "official_like_cot_fewshot_v2",
        "source_index": index, "category": str(row["category"]),
        "messages": [{"role": "user", "content": prompt}],
        "answer": str(row["answer"]).strip().upper(),
        "score_type": "multiple_choice", "max_tokens": MAX_OUTPUT_TOKENS,
        "sampling": {**SAMPLING, "seed": seed},
    }


def ceval_question(row: dict[str, Any]) -> str:
    choices = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    return f"问题：{row['question']}\n{choices}"


def ceval_record(
    row: dict[str, Any], subject: str, index: int,
    demonstrations: list[dict[str, Any]], seed: int,
) -> dict[str, Any]:
    prompt = (
        f"以下是关于 {subject} 的单项选择题（含答案示例）。请逐步思考，"
        '最后仅输出 JSON，例如 {"answer": "C"}。\n\n'
    )
    for example in demonstrations:
        prompt += ceval_question(example) + f"\n答案：{example['answer']}\n\n"
    prompt += ceval_question(row)
    return {
        "id": f"ceval:{subject}:val:{row.get('id', index)}",
        "benchmark": "ceval", "protocol": "official_like_5shot_v2",
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

    rows: list[dict[str, Any]] = []
    for index in stratified_indices(
        mmlu_test, 64, "category", args.seed, "official_like:mmlu_pro"
    ):
        row = mmlu_test[index]
        rows.append(mmlu_record(
            row, index, mmlu_demos[str(row["category"])], args.seed
        ))

    ceval_val_paths = sorted((raw / "ceval_val").glob("*.parquet"))
    ceval_dev_paths = sorted((raw / "ceval_dev").glob("*.parquet"))
    if len(ceval_val_paths) != 52 or len(ceval_dev_paths) != 52:
        raise RuntimeError("C-Eval requires 52 val and 52 dev subject files")
    dev_by_subject = {
        path.stem: pq.read_table(path).to_pylist() for path in ceval_dev_paths
    }
    for path in ceval_val_paths:
        subject = path.stem
        subject_rows = pq.read_table(path).to_pylist()
        [index] = sample_indices(
            len(subject_rows), 1, args.seed, f"official_like:ceval:{subject}"
        )
        rows.append(ceval_record(
            subject_rows[index], subject, index, dev_by_subject[subject], args.seed
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

    manifest = {
        "schema_version": "qtopomoe.official_like_smoke.v2",
        "path": str(args.output.resolve()), "records": len(rows),
        "counts": {"mmlu_pro": 64, "ceval": 52}, "seed": args.seed,
        "revisions": {
            "mmlu_pro": MMLU_REVISION, "ceval_main": CEVAL_MAIN_REVISION,
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
            "Protocol-aligned sampled reproduction with the model-card recommended "
            "32,768 output-token ceiling; not an official full-set score."
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
