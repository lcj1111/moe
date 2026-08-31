#!/usr/bin/env python3
# 作用：在共同可评分样本上比较两个 full-set 合并结果。
"""在共同可评分 ID 上比较两个 full-set 合并结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
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


def compare_many(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """在所有格式均可评分的 ID 上比较两个或更多 full-set 结果。"""

    if len(datasets) < 2:
        raise ValueError("多格式比较至少需要两个输入")
    labels = list(datasets)
    indexed = {
        label: index_unique(rows, label) for label, rows in datasets.items()
    }
    first_label = labels[0]
    ordered_ids = [str(row["id"]) for row in datasets[first_label]]
    expected_ids = set(indexed[first_label])
    for label in labels[1:]:
        if set(indexed[label]) != expected_ids:
            raise ValueError(f"{label} 与 {first_label} 的 ID 集合不一致")
        for row_id in ordered_ids:
            for field in ("id", "benchmark", "expected", "score_type"):
                if indexed[first_label][row_id].get(field) != indexed[label][row_id].get(field):
                    raise ValueError(f"{row_id} 的 {field} 在 {first_label}/{label} 间不一致")

    common_ids = [
        row_id
        for row_id in ordered_ids
        if all(indexed[label][row_id].get("correct") is not None for label in labels)
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id in common_ids:
        groups[str(indexed[first_label][row_id].get("benchmark", "unknown"))].append(row_id)

    def common_stats(group_ids: list[str]) -> dict[str, Any]:
        count = len(group_ids)
        formats: dict[str, dict[str, Any]] = {}
        correct_counts: dict[str, int] = {}
        for label in labels:
            correct = sum(
                indexed[label][row_id].get("correct") is True for row_id in group_ids
            )
            correct_counts[label] = correct
            formats[label] = {
                "correct": correct,
                "accuracy": correct / count if count else None,
            }
        deltas = {
            f"{right}_minus_{left}_percentage_points": (
                100.0 * (correct_counts[right] - correct_counts[left]) / count
                if count
                else None
            )
            for left, right in combinations(labels, 2)
        }
        return {
            "common_scored": count,
            "formats": formats,
            "pairwise_deltas": deltas,
        }

    patterns: Counter[str] = Counter()
    for row_id in common_ids:
        signature = "|".join(
            f"{label}={'correct' if indexed[label][row_id].get('correct') is True else 'incorrect'}"
            for label in labels
        )
        patterns[signature] += 1

    unfinished_sets = {
        label: {
            row_id
            for row_id in ordered_ids
            if indexed[label][row_id].get("correct") is None
        }
        for label in labels
    }
    unfinished_union = set().union(*unfinished_sets.values())
    unfinished_intersection = set.intersection(*unfinished_sets.values())
    return {
        "records": len(ordered_ids),
        "labels": labels,
        "own_scored_denominator": {
            label: stats(datasets[label]) for label in labels
        },
        "common_denominator": {
            "overall": common_stats(common_ids),
            "benchmarks": {
                benchmark: common_stats(group_ids)
                for benchmark, group_ids in sorted(groups.items())
            },
            "correctness_patterns": dict(sorted(patterns.items())),
        },
        "unfinished": {
            "by_format": {
                label: len(unfinished_sets[label]) for label in labels
            },
            "intersection_all_formats": len(unfinished_intersection),
            "union_all_formats": len(unfinished_union),
            "common_scored_expected": len(ordered_ids) - len(unfinished_union),
        },
        "checks": {
            "all_id_sets_equal": all(
                set(indexed[label]) == expected_ids for label in labels
            ),
            "all_ids_unique": all(
                len(indexed[label]) == len(datasets[label]) for label in labels
            ),
            "common_denominator_consistent": len(common_ids)
            == len(ordered_ids) - len(unfinished_union),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="标签=JSONL路径",
        help="可重复指定两个或更多格式；启用时不再使用 left/right 参数。",
    )
    parser.add_argument("--left", type=pathlib.Path)
    parser.add_argument("--right", type=pathlib.Path)
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.input:
        if any((args.left, args.right, args.left_label, args.right_label)):
            parser.error("--input 不能与 left/right 参数混用")
        inputs: dict[str, pathlib.Path] = {}
        for item in args.input:
            if "=" not in item:
                parser.error(f"--input 必须使用 标签=路径：{item}")
            label, raw_path = item.split("=", 1)
            if not label or not raw_path or label in inputs:
                parser.error(f"--input 标签/路径无效或标签重复：{item}")
            inputs[label] = pathlib.Path(raw_path)
        if len(inputs) < 2:
            parser.error("至少需要两个 --input")
        payload = {
            "schema_version": "qtopomoe.fullset_quality_comparison.v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in inputs.items()
            },
            "comparison": compare_many(
                {label: read_jsonl(path) for label, path in inputs.items()}
            ),
            "interpretation": {
                "primary_denominator": "all_formats_common_scored_ids",
                "reason": "各格式显式未完成 ID 不完全相同；全格式共同分母避免选择性缺失偏差。",
            },
        }
    else:
        if not all((args.left, args.right, args.left_label, args.right_label)):
            parser.error("必须提供完整 left/right 参数，或重复使用 --input")
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
