#!/usr/bin/env python3
"""从基础轮结果生成只包含截断样本的确定性续跑 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any


IDENTITY_FIELDS = (
    ("benchmark", "benchmark"),
    ("score_type", "score_type"),
    ("answer", "expected"),
    ("max_tokens", "max_tokens"),
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def validate_base_results(
    manifest: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    """在提取截断项前验证基础轮与冻结 manifest 完全一致。"""

    manifest_ids = [row["id"] for row in manifest]
    result_ids = [row["id"] for row in results]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("base manifest contains duplicate ids")
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("base results contain duplicate ids")
    if result_ids != manifest_ids:
        missing = sorted(set(manifest_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(manifest_ids))
        raise ValueError(
            "base result ids/order do not match manifest: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    for manifest_row, result_row in zip(manifest, results, strict=True):
        row_id = manifest_row["id"]
        for manifest_key, result_key in IDENTITY_FIELDS:
            if manifest_row.get(manifest_key) != result_row.get(result_key):
                raise ValueError(
                    f"identity mismatch for {row_id}: "
                    f"manifest.{manifest_key}={manifest_row.get(manifest_key)!r} "
                    f"result.{result_key}={result_row.get(result_key)!r}"
                )
        if result_row.get("error") is not None:
            raise ValueError(f"base result contains request failure: {row_id}")
        if result_row.get("truncated"):
            if result_row.get("correct") is not None:
                raise ValueError(f"truncated result must keep correct=null: {row_id}")
            usage = result_row.get("usage") or {}
            completion_tokens = usage.get(
                "completion_tokens", usage.get("output_tokens")
            )
            reached_token_limit = (
                completion_tokens is not None
                and int(completion_tokens) >= int(manifest_row["max_tokens"])
            )
            if result_row.get("finish_reason") != "length" and not reached_token_limit:
                raise ValueError(
                    "truncated result lacks length evidence "
                    f"(finish_reason or completion_tokens): {row_id}"
                )
        elif not isinstance(result_row.get("correct"), bool):
            raise ValueError(f"completed result must have boolean correct: {row_id}")


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
    validate_base_results(manifest, results)
    by_id = {row["id"]: row for row in results}

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
        "checks": {
            "manifest_and_results_same_length": len(manifest) == len(results),
            "ids_unique_and_ordered": True,
            "identity_fields_match": True,
            "request_failures_zero": True,
            "truncated_correct_is_null": True,
            "truncated_has_length_evidence": True,
        },
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
