#!/usr/bin/env python3
"""Freeze deterministic quality-smoke and formal evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
from collections import defaultdict
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "mmlu_pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "ceval_main": "617524a00b307ff6f9933702f724131fe12ca7ce",
    "ceval_parquet": "8267189d6ba0d516d414a98919558032958c4466",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
}
LETTERS = "ABCDEFGHIJ"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_indices(total: int, count: int, seed: int, label: str) -> list[int]:
    if count > total:
        raise ValueError(f"{label}: requested {count}, only {total} available")
    rng = random.Random(f"{seed}:{label}")
    return sorted(rng.sample(range(total), count))


def stratified_indices(
    rows: list[dict[str, Any]], count: int, key: str, seed: int, label: str
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row[key])].append(index)
    rng = random.Random(f"{seed}:{label}")
    for indices in groups.values():
        rng.shuffle(indices)
    selected: list[int] = []
    names = sorted(groups)
    cursor = 0
    while len(selected) < count:
        made_progress = False
        for name in names:
            if cursor < len(groups[name]):
                selected.append(groups[name][cursor])
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise ValueError(f"{label}: unable to select {count} rows")
        cursor += 1
    return sorted(selected)


def gsm_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    answer = str(row["answer"]).rsplit("####", 1)[-1].strip()
    prompt = (
        "Solve the following math problem. Show concise reasoning and finish with "
        "exactly `FINAL: <number>`.\n\n" + str(row["question"])
    )
    return {
        "id": f"gsm8k:test:{index}", "benchmark": "gsm8k", "source_index": index,
        "messages": [{"role": "user", "content": prompt}],
        "answer": answer, "score_type": "numeric", "max_tokens": 512,
    }


def mmlu_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    options = list(row["options"])
    body = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))
    prompt = (
        "Choose the single best answer. Respond with exactly `FINAL: <letter>` "
        "and no explanation.\n\n" + str(row["question"]) + "\n" + body
    )
    return {
        "id": f"mmlu_pro:test:{row.get('question_id', index)}",
        "benchmark": "mmlu_pro", "source_index": index,
        "category": str(row["category"]),
        "messages": [{"role": "user", "content": prompt}],
        "answer": str(row["answer"]).strip().upper(),
        "score_type": "multiple_choice", "max_tokens": 32,
    }


def ceval_record(row: dict[str, Any], subject: str, index: int) -> dict[str, Any]:
    body = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    prompt = (
        "请选择唯一正确答案。只输出 `FINAL: <选项字母>`，不要解释。\n\n"
        + str(row["question"]) + "\n" + body
    )
    return {
        "id": f"ceval:{subject}:val:{row.get('id', index)}",
        "benchmark": "ceval", "source_index": index, "subject": subject,
        "messages": [{"role": "user", "content": prompt}],
        "answer": str(row["answer"]).strip().upper(),
        "score_type": "multiple_choice", "max_tokens": 32,
    }


def humaneval_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = (
        "Complete the following Python function. Return only the code continuation, "
        "without Markdown fences.\n\n" + str(row["prompt"])
    )
    return {
        "id": str(row["task_id"]), "benchmark": "humaneval",
        "source_index": index,
        "messages": [{"role": "user", "content": prompt}],
        "entry_point": str(row["entry_point"]),
        "score_type": "deferred_code", "max_tokens": 512,
    }


def token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def validate_tokenizers(
    rows: list[dict[str, Any]], tokenizer_a_path: pathlib.Path,
    tokenizer_b_path: pathlib.Path,
) -> str:
    tokenizer_a = AutoTokenizer.from_pretrained(tokenizer_a_path, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(tokenizer_b_path, trust_remote_code=True)
    digest = hashlib.sha256()
    for row in rows:
        ids_a = token_ids(tokenizer_a, row["messages"])
        ids_b = token_ids(tokenizer_b, row["messages"])
        if ids_a != ids_b:
            raise RuntimeError(f"tokenizer/chat-template mismatch for {row['id']}")
        digest.update(json.dumps(ids_a, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-a", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-b", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.raw_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    gsm = pq.read_table(raw / "gsm8k_test.parquet").to_pylist()
    mmlu = pq.read_table(raw / "mmlupro_test.parquet").to_pylist()
    humaneval = pq.read_table(raw / "humaneval_test.parquet").to_pylist()
    ceval_by_subject = {
        path.stem: pq.read_table(path).to_pylist()
        for path in sorted((raw / "ceval_val").glob("*.parquet"))
    }
    if len(ceval_by_subject) != 52:
        raise SystemExit(f"expected 52 C-Eval subjects, got {len(ceval_by_subject)}")

    def build_set(name: str, gsm_n: int, mmlu_n: int, ceval_per: int, human_n: int):
        rows: list[dict[str, Any]] = []
        for index in sample_indices(len(gsm), gsm_n, args.seed, f"{name}:gsm8k"):
            rows.append(gsm_record(gsm[index], index))
        for index in stratified_indices(
            mmlu, mmlu_n, "category", args.seed, f"{name}:mmlu_pro"
        ):
            rows.append(mmlu_record(mmlu[index], index))
        for subject, subject_rows in sorted(ceval_by_subject.items()):
            count = min(ceval_per, len(subject_rows))
            indices = sample_indices(
                len(subject_rows), count, args.seed, f"{name}:ceval:{subject}"
            )
            rows.extend(ceval_record(subject_rows[i], subject, i) for i in indices)
        for index in sample_indices(
            len(humaneval), human_n, args.seed, f"{name}:humaneval"
        ):
            rows.append(humaneval_record(humaneval[index], index))
        return rows

    sets = {
        "quality_smoke_v1": build_set("smoke", 32, 64, 1, 16),
        "quality_formal_v1": build_set("formal", 256, 512, 10, 64),
    }
    raw_files = [path for path in sorted(raw.rglob("*.parquet"))]
    manifest: dict[str, Any] = {
        "schema_version": "qtopomoe.quality_sets.v1", "seed": args.seed,
        "revisions": REVISIONS,
        "raw_files": [
            {"path": str(path.relative_to(raw)), "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in raw_files
        ],
        "sets": {},
        "tokenizers": {"a": str(args.tokenizer_a.resolve()),
                       "b": str(args.tokenizer_b.resolve())},
    }
    for name, rows in sets.items():
        path = output / f"{name}.jsonl"
        token_sha = validate_tokenizers(rows, args.tokenizer_a, args.tokenizer_b)
        for row in rows:
            row["sampling"] = {"temperature": 0, "seed": args.seed,
                               "enable_thinking": False}
        write_jsonl(path, rows)
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["benchmark"]] += 1
        manifest["sets"][name] = {
            "path": str(path), "records": len(rows), "counts": dict(counts),
            "sha256": sha256_file(path), "prompt_token_ids_sha256": token_sha,
            "token_ids_equal": True,
        }
    manifest_path = output / "quality_sets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["sets"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
