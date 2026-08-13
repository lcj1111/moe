#!/usr/bin/env python3
"""从基础轮结果生成只包含截断样本的确定性续跑 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--base-results", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--answer-only-max-tokens", type=int, default=8192)
    parser.add_argument("--cot-max-tokens", type=int, default=12000)
    args = parser.parse_args()

    manifest = read_jsonl(args.base_manifest)
    results = read_jsonl(args.base_results)
    by_id = {row["id"]: row for row in results}
    if len(by_id) != len(results):
        raise ValueError("base results contain duplicate ids")

    selected: list[dict[str, Any]] = []
    original_limits: Counter[int] = Counter()
    updated_limits: Counter[int] = Counter()
    protocols: Counter[str] = Counter()
    for row in manifest:
        result = by_id.get(row["id"])
        if result is None or not result.get("truncated"):
            continue
        updated = dict(row)
        protocol = str(updated.get("protocol", ""))
        new_limit = (
            args.cot_max_tokens
            if protocol == "official_cot_fewshot_full"
            else args.answer_only_max_tokens
        )
        old_limit = int(updated["max_tokens"])
        if new_limit <= old_limit:
            raise ValueError(f"new max_tokens must exceed old limit for {row['id']}")
        updated["max_tokens"] = new_limit
        selected.append(updated)
        original_limits[old_limit] += 1
        updated_limits[new_limit] += 1
        protocols[protocol] += 1

    expected = sum(bool(row.get("truncated")) for row in results)
    if len(selected) != expected:
        raise ValueError(f"selected={len(selected)} but truncated results={expected}")
    if not selected:
        raise ValueError("no truncated samples found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(body, encoding="utf-8", newline="\n")
    temporary.replace(args.output)

    metadata = {
        "schema_version": "qtopomoe.fullset_truncation_manifest.v1",
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha256(args.base_manifest),
        "base_results": str(args.base_results),
        "base_results_sha256": sha256(args.base_results),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "samples": len(selected),
        "protocol_counts": dict(sorted(protocols.items())),
        "original_max_tokens": {str(k): v for k, v in sorted(original_limits.items())},
        "rerun_max_tokens": {str(k): v for k, v in sorted(updated_limits.items())},
    }
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
