#!/usr/bin/env python3
"""严格合并 full-set 基础轮与截断续跑结果，并生成可审计 Gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"{path}:{line_number} 缺少非空 id")
            rows.append(row)
    return rows


def read_many(paths: Iterable[pathlib.Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
        hashes[str(path)] = sha256(path)
    return rows, hashes


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in indexed:
            duplicates.append(row_id)
        else:
            indexed[row_id] = row
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"{label} 含重复 id：{preview}")
    return indexed


def is_failed(row: dict[str, Any]) -> bool:
    return bool(row.get("error")) or row.get("failed") is True


def validate_identity(base: dict[str, Any], rerun: dict[str, Any]) -> None:
    for field in ("id", "benchmark", "expected", "score_type"):
        if base.get(field) != rerun.get(field):
            raise ValueError(
                f"续跑记录 {base.get('id')} 的 {field} 与基础轮不一致："
                f"{base.get(field)!r} != {rerun.get(field)!r}"
            )


def benchmark_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("benchmark", "unknown"))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for benchmark, items in sorted(grouped.items()):
        scored = [row for row in items if row.get("correct") is not None]
        correct = sum(row.get("correct") is True for row in scored)
        result[benchmark] = {
            "records": len(items),
            "scored": len(scored),
            "correct": correct,
            "accuracy_scored_only": (correct / len(scored)) if scored else None,
            "failed": sum(is_failed(row) for row in items),
            "truncated": sum(bool(row.get("truncated")) for row in items),
        }
    return result


def write_jsonl_atomic(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def merge_results(
    manifest_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_id = index_unique(manifest_rows, "manifest")
    base_by_id = index_unique(base_rows, "基础轮结果")
    rerun_by_id = index_unique(rerun_rows, "续跑结果")
    manifest_ids = set(manifest_by_id)
    base_ids = set(base_by_id)
    rerun_ids = set(rerun_by_id)

    if base_ids != manifest_ids:
        missing = sorted(manifest_ids - base_ids)[:10]
        extra = sorted(base_ids - manifest_ids)[:10]
        raise ValueError(f"基础轮 id 与 manifest 不一致；missing={missing}, extra={extra}")

    base_truncated_ids = {
        row_id for row_id, row in base_by_id.items() if bool(row.get("truncated"))
    }
    if rerun_ids != base_truncated_ids:
        missing = sorted(base_truncated_ids - rerun_ids)[:10]
        extra = sorted(rerun_ids - base_truncated_ids)[:10]
        raise ValueError(
            "续跑 id 必须与基础轮截断 id 完全一致；"
            f"missing={missing}, extra={extra}"
        )

    merged: list[dict[str, Any]] = []
    sources: Counter[str] = Counter()
    for manifest_row in manifest_rows:
        row_id = str(manifest_row["id"])
        base = base_by_id[row_id]
        if row_id in rerun_by_id:
            selected = rerun_by_id[row_id]
            validate_identity(base, selected)
            source = "rerun"
        else:
            selected = base
            source = "base"
        output_row = dict(selected)
        output_row["merge_source"] = source
        output_row["base_was_truncated"] = bool(base.get("truncated"))
        merged.append(output_row)
        sources[source] += 1

    merged_ids = [str(row["id"]) for row in merged]
    expected_order = [str(row["id"]) for row in manifest_rows]
    residual = [row for row in merged if bool(row.get("truncated"))]
    failed = [row for row in merged if is_failed(row)]
    scored = [row for row in merged if row.get("correct") is not None]
    correct = sum(row.get("correct") is True for row in scored)
    audit = {
        "manifest_records": len(manifest_rows),
        "base_records": len(base_rows),
        "base_truncated": len(base_truncated_ids),
        "rerun_records": len(rerun_rows),
        "replaced_records": sources["rerun"],
        "source_counts": dict(sorted(sources.items())),
        "merged_records": len(merged),
        "merged_unique_ids": len(set(merged_ids)),
        "failed": len(failed),
        "truncated": len(residual),
        "scored": len(scored),
        "correct": correct,
        "accuracy_scored_only": (correct / len(scored)) if scored else None,
        "benchmark_stats": benchmark_stats(merged),
        "residual_truncated_ids": [str(row["id"]) for row in residual],
        "checks": {
            "base_exactly_matches_manifest": base_ids == manifest_ids,
            "rerun_exactly_matches_base_truncated": rerun_ids == base_truncated_ids,
            "merged_ids_unique": len(merged_ids) == len(set(merged_ids)),
            "merged_order_matches_manifest": merged_ids == expected_order,
            "merged_count_matches_manifest": len(merged) == len(manifest_rows),
            "failed_is_zero": not failed,
        },
    }
    return merged, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--base-results", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--rerun-results", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--format", required=True)
    args = parser.parse_args()

    manifest_rows, manifest_hashes = read_many(args.manifest)
    base_rows, base_hashes = read_many(args.base_results)
    rerun_rows, rerun_hashes = read_many(args.rerun_results)
    merged, audit = merge_results(manifest_rows, base_rows, rerun_rows)
    write_jsonl_atomic(args.output, merged)

    all_integrity_checks = all(audit["checks"].values())
    if not all_integrity_checks:
        raise ValueError("合并后完整性检查失败")
    if audit["truncated"]:
        overall = "closed_with_unfinished"
        completion_gate = "incomplete_explicit"
    else:
        overall = "accepted"
        completion_gate = "accepted"
    summary = {
        "schema_version": "qtopomoe.fullset_quality_merge.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "inputs": {
            "manifest": {"paths": [str(path) for path in args.manifest], "sha256": manifest_hashes},
            "base_results": {"paths": [str(path) for path in args.base_results], "sha256": base_hashes},
            "rerun_results": {"paths": [str(path) for path in args.rerun_results], "sha256": rerun_hashes},
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
        },
        "audit": audit,
        "gate": {
            "overall": overall,
            "integrity_gate": "accepted",
            "request_gate": "accepted" if audit["failed"] == 0 else "rejected",
            "completion_gate": completion_gate,
            "accuracy_denominator": "scored_only",
            "unfinished_policy": "截断记录 correct 必须为 null，不计入正确或错误；保留 id 供后续研究。",
        },
    }
    write_json_atomic(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
