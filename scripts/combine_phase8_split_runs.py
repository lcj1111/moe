#!/usr/bin/env python3
"""无损合并拆分执行的 Phase 8 正式重复实验。

输出目录只保存合并后的 schedule、来源清单和指向原始 run 目录的符号链接；
不会复制或修改任何原始实验结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workload-matrix", type=Path, required=True)
    args = parser.parse_args()

    sources = [path.resolve() for path in args.source_root]
    matrix = args.workload_matrix.resolve()
    if not matrix.is_file():
        raise SystemExit(f"workload matrix not found: {matrix}")

    schedules: list[tuple[Path, Path, dict[str, Any]]] = []
    seeds: set[int] = set()
    matrix_hashes: set[str] = set()
    merged_items: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    manifest_sources: list[dict[str, Any]] = []

    for root in sources:
        schedule_path = root / "schedule.json"
        schedule = load_json(schedule_path)
        schedules.append((root, schedule_path, schedule))
        seeds.add(int(schedule["seed"]))
        matrix_hashes.add(str(schedule["workload_matrix_sha256"]))
        manifest_sources.append({
            "root": str(root),
            "schedule": str(schedule_path),
            "schedule_sha256": sha256(schedule_path),
            "run_count": len(schedule["schedule"]),
        })
        for item in schedule["schedule"]:
            run_id = f"{item['candidate_id']}__r{int(item['repeat']):02d}"
            if run_id in seen_run_ids:
                raise SystemExit(f"duplicate run id: {run_id}")
            run_dir = root / "runs" / run_id
            if not (run_dir / "meta.json").is_file():
                raise SystemExit(f"missing run metadata: {run_dir / 'meta.json'}")
            seen_run_ids.add(run_id)
            merged_items.append({
                "candidate_id": item["candidate_id"],
                "repeat": int(item["repeat"]),
                "source_run": str(run_dir),
            })

    if len(seeds) != 1:
        raise SystemExit(f"source seeds differ: {sorted(seeds)}")
    actual_matrix_hash = sha256(matrix)
    if matrix_hashes != {actual_matrix_hash}:
        raise SystemExit(
            "workload matrix hash mismatch: "
            f"sources={sorted(matrix_hashes)}, actual={actual_matrix_hash}"
        )

    merged_items.sort(key=lambda row: (row["candidate_id"], row["repeat"]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = args.output_root / "runs"
    runs_root.mkdir(exist_ok=True)
    schedule_items = []
    for order, item in enumerate(merged_items, start=1):
        run_id = f"{item['candidate_id']}__r{item['repeat']:02d}"
        link = runs_root / run_id
        target = Path(item["source_run"])
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise SystemExit(f"existing link has different target: {link}")
        elif link.exists():
            raise SystemExit(f"refusing to replace existing path: {link}")
        else:
            os.symlink(target, link, target_is_directory=True)
        schedule_items.append({
            "candidate_id": item["candidate_id"],
            "order": order,
            "repeat": item["repeat"],
        })

    first = schedules[0][2]
    merged_schedule = {
        "schema_version": "qtopomoe.phase8_combined_schedule.v1",
        "seed": next(iter(seeds)),
        "workload_matrix": str(matrix),
        "workload_matrix_sha256": actual_matrix_hash,
        "implementation": first.get("implementation"),
        "runtime": first.get("runtime"),
        "schedule": schedule_items,
    }
    schedule_out = args.output_root / "schedule.json"
    schedule_out.write_text(
        json.dumps(merged_schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "qtopomoe.phase8_split_merge_manifest.v1",
        "method": "symbolic_links_to_immutable_source_runs",
        "source_roots": manifest_sources,
        "workload_matrix": str(matrix),
        "workload_matrix_sha256": actual_matrix_hash,
        "combined_schedule": str(schedule_out),
        "combined_schedule_sha256": sha256(schedule_out),
        "candidate_count": len({row["candidate_id"] for row in schedule_items}),
        "run_count": len(schedule_items),
    }
    manifest_out = args.output_root / "merge_manifest.json"
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
