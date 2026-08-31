#!/usr/bin/env python3
# 作用：从容量预跑中冻结候选共享的公平到达率。
"""Audit a controlled capacity prepass and freeze fair Phase 8 arrival rates.

The formal open-loop matrix must not offer a different load to each strategy.
For every base workload cell this tool takes the minimum observed closed-loop
request throughput across every admitted candidate and repeat, applies a
documented safety utilization, and assigns the resulting common rate to both
Poisson and burst arrivals.  Closed-loop cells remain rate-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def stable_stream_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}/{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def floor_rate(value: float, digits: int = 6) -> float:
    scale = 10 ** digits
    return math.floor(value * scale) / scale


def audit_capacity(run_root: Path, expected_repeats: int) -> dict[str, Any]:
    schedule_path = run_root / "schedule.json"
    if not schedule_path.exists():
        raise ValueError(f"missing schedule: {schedule_path}")
    schedule = load_json(schedule_path)
    matrix_path = Path(schedule["workload_matrix"])
    matrix = load_json(matrix_path)
    cells = {row["id"]: row for row in matrix["cells"]}
    errors: list[str] = []
    observations: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    candidates = sorted({row["candidate_id"] for row in schedule["schedule"]})

    for cell_id, cell in cells.items():
        if cell.get("prefix_cache_pct") != 0:
            errors.append(f"{cell_id}: capacity prepass prefix_cache_pct must be 0")
        if cell.get("arrival_mode") != "closed_loop":
            errors.append(f"{cell_id}: capacity prepass arrival_mode must be closed_loop")

    for item in schedule["schedule"]:
        candidate_id = item["candidate_id"]
        repeat = int(item["repeat"])
        run_id = f"{candidate_id}__r{repeat:02d}"
        run_dir = run_root / "runs" / run_id
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            errors.append(f"{run_id}: missing meta")
            continue
        meta = load_json(meta_path)
        if meta.get("status") != "workload_passed":
            errors.append(f"{run_id}: status={meta.get('status')}")
        for cell_id, cell in cells.items():
            summary_path = run_dir / "workloads" / cell_id / "summary.json"
            if not summary_path.exists():
                errors.append(f"{run_id}/{cell_id}: missing summary")
                continue
            summary = load_json(summary_path)
            checks = {
                "failed_zero": summary.get("failed") == 0,
                "complete": summary.get("completed") == summary.get("requests"),
                "tokens_exact": summary.get("input_tokens_actual") == cell["input_tokens"]
                                and summary.get("server_prompt_tokens_exact") is True,
                "concurrency": summary.get("concurrency") == cell["concurrency"],
                "prefix_zero": summary.get("prefix_cache", {}).get("target_pct") == 0,
                "cache_zero": summary.get("prefix_cache", {}).get(
                    "actual_cached_token_ratio") == 0,
                "cache_gate": summary.get("prefix_cache", {}).get("ratio_gate") is True,
                "cache_details": summary.get("prefix_cache", {}).get(
                    "usage_details_complete") is True,
                "closed_loop": summary.get("arrival", {}).get("mode") == "closed_loop",
                "arrival_gate": summary.get("arrival", {}).get("schedule_gate") is True,
            }
            if not all(checks.values()):
                errors.append(f"{run_id}/{cell_id}: Gate={checks}")
                continue
            wall_time = summary.get("wall_time_s")
            completed = summary.get("completed")
            if not isinstance(wall_time, (int, float)) or wall_time <= 0:
                errors.append(f"{run_id}/{cell_id}: invalid wall_time_s={wall_time}")
                continue
            request_rps = float(completed) / float(wall_time)
            observations[cell_id][candidate_id].append({
                "repeat": repeat,
                "request_rps": request_rps,
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
            })

    cell_results = []
    for cell_id, cell in cells.items():
        candidate_results = []
        for candidate_id in candidates:
            rows = sorted(observations[cell_id][candidate_id],
                          key=lambda row: row["repeat"])
            if len(rows) != expected_repeats:
                errors.append(
                    f"{candidate_id}/{cell_id}: repeats={len(rows)} "
                    f"expected={expected_repeats}")
            values = [row["request_rps"] for row in rows]
            candidate_results.append({
                "candidate_id": candidate_id,
                "repeat_request_rps": values,
                "median_request_rps": statistics.median(values) if values else None,
                "minimum_request_rps": min(values) if values else None,
                "artifacts": rows,
            })
        admitted = [row["minimum_request_rps"] for row in candidate_results
                    if isinstance(row["minimum_request_rps"], (int, float))]
        cell_results.append({
            "cell_id": cell_id,
            "input_tokens": cell["input_tokens"],
            "output_tokens": cell["output_tokens"],
            "concurrency": cell["concurrency"],
            "requests": cell["requests"],
            "minimum_observed_request_rps": min(admitted) if admitted else None,
            "candidates": candidate_results,
        })

    return {
        "schema_version": "qtopomoe.phase8_capacity_audit.v1",
        "status": "accepted" if not errors else "rejected",
        "rate_basis": "minimum observed closed-loop request/s across all candidates and repeats",
        "expected_repeats": expected_repeats,
        "schedule": str(schedule_path),
        "schedule_sha256": sha256(schedule_path),
        "workload_matrix": str(matrix_path),
        "workload_matrix_sha256": sha256(matrix_path),
        "candidate_ids": candidates,
        "cells": cell_results,
        "errors": errors,
    }


def build_formal_matrix(audit: dict[str, Any], safety_utilization: float,
                        seed: int, audit_path: Path) -> dict[str, Any]:
    if audit.get("status") != "accepted":
        raise ValueError("capacity audit is not accepted")
    if not 0 < safety_utilization < 1:
        raise ValueError("safety utilization must be between 0 and 1")
    cells = []
    rate_freeze = []
    for base in audit["cells"]:
        capacity = base["minimum_observed_request_rps"]
        if not isinstance(capacity, (int, float)) or capacity <= 0:
            raise ValueError(f"invalid capacity for {base['cell_id']}: {capacity}")
        frozen_rate = floor_rate(float(capacity) * safety_utilization)
        if frozen_rate <= 0:
            raise ValueError(f"frozen rate is not positive for {base['cell_id']}")
        rate_freeze.append({
            "base_cell_id": base["cell_id"],
            "minimum_observed_request_rps": capacity,
            "safety_utilization": safety_utilization,
            "frozen_request_rate_rps": frozen_rate,
        })
        for prefix in (0, 50, 100):
            for arrival in ("closed_loop", "poisson", "burst"):
                cell_id = f"{base['cell_id']}_p{prefix}_{arrival}"
                row = {
                    "id": cell_id,
                    "base_cell_id": base["cell_id"],
                    "input_tokens": base["input_tokens"],
                    "output_tokens": base["output_tokens"],
                    "concurrency": base["concurrency"],
                    "requests": base["requests"],
                    "prefix_cache_pct": prefix,
                    "arrival_mode": arrival,
                    "stream_seed": stable_stream_seed(seed, cell_id),
                }
                if arrival != "closed_loop":
                    row["request_rate_rps"] = frozen_rate
                cells.append(row)
    return {
        "schema_version": "qtopomoe.workload_matrix.v3",
        "purpose": "Formal controlled Phase 8 matrix with candidate-independent arrival rates",
        "seed": seed,
        "capacity_audit": str(audit_path),
        "capacity_audit_sha256": sha256(audit_path),
        "rate_policy": {
            "basis": audit["rate_basis"],
            "safety_utilization": safety_utilization,
            "rounding": "floor to 6 decimal places",
            "same_rate_for_poisson_and_burst": True,
            "closed_loop_is_rate_free": True,
        },
        "rate_freeze": rate_freeze,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-matrix", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--safety-utilization", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit = audit_capacity(args.run_root, args.expected_repeats)
    write_json(args.output_audit, audit)
    if audit["status"] != "accepted":
        print(json.dumps({"status": "rejected", "errors": audit["errors"]}))
        raise SystemExit(1)
    matrix = build_formal_matrix(audit, args.safety_utilization,
                                 args.seed, args.output_audit)
    write_json(args.output_matrix, matrix)
    print(json.dumps({
        "status": "accepted",
        "base_cells": len(audit["cells"]),
        "formal_cells": len(matrix["cells"]),
        "output_audit": str(args.output_audit),
        "output_matrix": str(args.output_matrix),
    }))


if __name__ == "__main__":
    main()
