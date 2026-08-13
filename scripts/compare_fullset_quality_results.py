#!/usr/bin/env python3
"""在共同可评分 ID 上比较两个 full-set 合并结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} 不是合法 JSON") from exc
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"{path}:{number} 缺少非空 id")
            rows.append(row)
    return rows


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        if row_id in result:
            raise ValueError(f"{label} 含重复 id：{row_id}")
        result[row_id] = row
    return result


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("correct") is not None]
    correct = sum(row.get("correct") is True for row in scored)
    return {
        "records": len(rows),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "unfinished": len(rows) - len(scored),
    }


def compare(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left = index_unique(left_rows, left_label)
    right = index_unique(right_rows, right_label)
    if set(left) != set(right):
        raise ValueError("两侧结果 ID 集合不一致")

    ids = [str(row["id"]) for row in left_rows]
    for row_id in ids:
        for field in ("id", "benchmark", "expected", "score_type"):
            if left[row_id].get(field) != right[row_id].get(field):
                raise ValueError(f"{row_id} 的 {field} 不一致")

    common_ids = [
        row_id
        for row_id in ids
        if left[row_id].get("correct") is not None
        and right[row_id].get("correct") is not None
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id in common_ids:
        groups[str(left[row_id].get("benchmark", "unknown"))].append(row_id)

    def common_stats(group_ids: list[str]) -> dict[str, Any]:
        left_correct = sum(left[row_id].get("correct") is True for row_id in group_ids)
        right_correct = sum(right[row_id].get("correct") is True for row_id in group_ids)
        count = len(group_ids)
        return {
            "common_scored": count,
            left_label: {
                "correct": left_correct,
                "accuracy": left_correct / count if count else None,
            },
            right_label: {
                "correct": right_correct,
                "accuracy": right_correct / count if count else None,
            },
            f"delta_percentage_points_{right_label}_minus_{left_label}": (
                100.0 * (right_correct - left_correct) / count if count else None
            ),
        }

    disagreements = {
        f"{left_label}_correct_{right_label}_incorrect": 0,
        f"{left_label}_incorrect_{right_label}_correct": 0,
        "both_correct": 0,
        "both_incorrect": 0,
    }
    for row_id in common_ids:
        pair = (left[row_id].get("correct") is True, right[row_id].get("correct") is True)
        if pair == (True, False):
            disagreements[f"{left_label}_correct_{right_label}_incorrect"] += 1
        elif pair == (False, True):
            disagreements[f"{left_label}_incorrect_{right_label}_correct"] += 1
        elif pair == (True, True):
            disagreements["both_correct"] += 1
        else:
            disagreements["both_incorrect"] += 1

    left_unfinished = {row_id for row_id in ids if left[row_id].get("correct") is None}
    right_unfinished = {row_id for row_id in ids if right[row_id].get("correct") is None}
    return {
        "records": len(ids),
        "own_scored_denominator": {
            left_label: stats(left_rows),
            right_label: stats(right_rows),
        },
        "common_denominator": {
            "overall": common_stats(common_ids),
            "benchmarks": {
                benchmark: common_stats(group_ids)
                for benchmark, group_ids in sorted(groups.items())
            },
            "disagreements": disagreements,
        },
        "unfinished": {
            left_label: len(left_unfinished),
            right_label: len(right_unfinished),
            "overlap": len(left_unfinished & right_unfinished),
            "union": len(left_unfinished | right_unfinished),
            "common_scored_expected": len(ids) - len(left_unfinished | right_unfinished),
        },
        "checks": {
            "id_sets_equal": set(left) == set(right),
            "ids_unique": len(left) == len(left_rows) and len(right) == len(right_rows),
            "common_denominator_consistent": len(common_ids)
            == len(ids) - len(left_unfinished | right_unfinished),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=pathlib.Path, required=True)
    parser.add_argument("--right", type=pathlib.Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    payload = {
        "schema_version": "qtopomoe.fullset_quality_comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            args.left_label: {"path": str(args.left), "sha256": sha256(args.left)},
            args.right_label: {"path": str(args.right), "sha256": sha256(args.right)},
        },
        "comparison": compare(
            read_jsonl(args.left),
            read_jsonl(args.right),
            args.left_label,
            args.right_label,
        ),
        "interpretation": {
            "primary_denominator": "common_scored_ids",
            "reason": "两种格式的显式未完成 ID 不完全相同；共同分母避免选择性缺失偏差。",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
