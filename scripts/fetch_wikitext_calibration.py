#!/usr/bin/env python3
"""Build a frozen 256-record calibration manifest from WikiText.

The source parquet is downloaded separately from a pinned Hugging Face
revision. This script records the exact source hash, row indices, tokenizer
hashes, chat template hash, and output hash so the calibration input can be
recreated without relying on a mutable `main` pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


DATASET_REPO = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tokens", type=int, default=2048)
    args = parser.parse_args()

    source = args.source_parquet.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"missing source parquet: {source}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    dataset = Dataset.from_parquet(str(source))
    if "text" not in dataset.column_names:
        raise SystemExit(f"source has no text column: {dataset.column_names}")

    candidates = [
        (index, str(value).strip())
        for index, value in enumerate(dataset["text"])
        if value is not None and str(value).strip()
    ]
    random.Random(args.seed).shuffle(candidates)

    records: list[dict] = []
    buffer: list[tuple[int, str]] = []
    buffer_tokens = 0
    for source_index, text in candidates:
        buffer.append((source_index, text))
        buffer_tokens += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if buffer_tokens < args.target_tokens:
            continue
        records.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "\n\n".join(item[1] for item in buffer),
                    }
                ],
                "source_row_indices": [item[0] for item in buffer],
                "source_token_count": buffer_tokens,
            }
        )
        if len(records) >= args.samples:
            break
        buffer = []
        buffer_tokens = 0

    if len(records) != args.samples:
        raise SystemExit(
            f"source produced {len(records)} records, requires {args.samples}"
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    template = tokenizer.chat_template or ""
    tokenizer_files = {}
    for name in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        path = tokenizer_path / name
        if path.is_file():
            tokenizer_files[name] = sha256(path)

    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "source_parquet_sha256": sha256(source),
        "source_row_count": len(dataset),
        "sample_count": len(records),
        "selection_seed": args.seed,
        "target_tokens_per_record": args.target_tokens,
        "sample_indices": [record["source_row_indices"] for record in records],
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_files_sha256": tokenizer_files,
        "chat_template_sha256": digest_text(template),
        "chat_template": template,
        "calibration_jsonl_sha256": sha256(output_jsonl),
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
