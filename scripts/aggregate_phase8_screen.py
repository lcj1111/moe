#!/usr/bin/env python3
"""Audit and compact Phase 8 single-pass service-screen artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_gpu_stats(root: Path, gpu_ids: set[int]) -> dict[str, float | None]:
    path = root / "gpu_memory.csv"
    if not path.exists():
        return {"peak_selected_gpu_memory_sum_mib": None,
                "peak_selected_gpu_power_sum_w": None}
    memory_samples: list[float] = []
    power_samples: list[float] = []
    seen_indices: set[int] = set()
    memory_sum = 0.0
    power_sum = 0.0
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5 or row[0].strip() == "timestamp":
                continue
            try:
                index = int(row[1].strip())
                if index not in gpu_ids:
                    continue
                if index in seen_indices:
                    memory_samples.append(memory_sum)
                    power_samples.append(power_sum)
                    seen_indices, memory_sum, power_sum = set(), 0.0, 0.0
                seen_indices.add(index)
                memory_sum += float(row[2].strip())
                power_sum += float(row[4].strip())
            except ValueError:
                continue
    if seen_indices:
        memory_samples.append(memory_sum)
        power_samples.append(power_sum)
    return {
        "peak_selected_gpu_memory_sum_mib": max(memory_samples, default=None),
        "peak_selected_gpu_power_sum_w": max(power_samples, default=None),
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Resource-aware Pareto dominance: lower latency/GPUs, higher throughput."""
    left_values = (left["median_e2e_p99_ms"], -left["median_output_tokens_s"],
                   left["gpu_count"])
    right_values = (right["median_e2e_p99_ms"], -right["median_output_tokens_s"],
                    right["gpu_count"])
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True,
                        help="candidate_id=/absolute/run/root; repeat roots to merge")
    parser.add_argument("--workload-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots_by_candidate: dict[str, list[Path]] = defaultdict(list)
    for spec in args.candidate:
        candidate_id, separator, raw_root = spec.partition("=")
        if not separator or not candidate_id or not raw_root:
            parser.error(f"invalid --candidate {spec!r}")
        roots_by_candidate[candidate_id].append(Path(raw_root))

    workload = json.loads(args.workload_matrix.read_text(encoding="utf-8"))
    expected = {row["id"]: row for row in workload["cells"]}
    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    candidate_meta: dict[str, dict[str, Any]] = {}
    root_audit: list[dict[str, Any]] = []
    errors: list[str] = []

    for candidate_id, roots in roots_by_candidate.items():
        gpu_ids: set[int] | None = None
        root_stats = []
        for root in roots:
            meta_path = root / "meta.json"
            if not meta_path.exists():
                errors.append(f"{candidate_id}: missing {meta_path}")
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            root_gpu_ids = {int(value) for value in str(meta["gpu_ids"]).split(",")}
            ep_enabled = bool(meta["ep"])
            expected_ep_ranks = int(meta.get("expected_ep_ranks", 0))
            actual_ep_ranks = int(meta.get("actual_ep_ranks", 0))
            if ep_enabled and (expected_ep_ranks <= 0 or
                               actual_ep_ranks != expected_ep_ranks):
                errors.append(
                    f"{candidate_id}: {root} EP ranks actual={actual_ep_ranks} "
                    f"expected={expected_ep_ranks}")
            if not ep_enabled and actual_ep_ranks != 0:
                errors.append(
                    f"{candidate_id}: {root} EP disabled but actual_ep_ranks="
                    f"{actual_ep_ranks}")
            if gpu_ids is None:
                gpu_ids = root_gpu_ids
                candidate_meta[candidate_id] = {
                    "model_path": meta["model_path"], "tp": meta["tp"],
                    "dp": meta["dp"], "ep_enabled": ep_enabled,
                    "expected_ep_ranks": expected_ep_ranks,
                    "actual_ep_ranks": actual_ep_ranks,
                    "gpu_ids": sorted(root_gpu_ids), "gpu_count": len(root_gpu_ids),
                }
            elif root_gpu_ids != gpu_ids:
                errors.append(f"{candidate_id}: inconsistent GPU IDs across roots")
            elif any((meta["model_path"] != candidate_meta[candidate_id]["model_path"],
                      meta["tp"] != candidate_meta[candidate_id]["tp"],
                      meta["dp"] != candidate_meta[candidate_id]["dp"],
                      ep_enabled != candidate_meta[candidate_id]["ep_enabled"],
                      expected_ep_ranks != candidate_meta[candidate_id]["expected_ep_ranks"],
                      actual_ep_ranks != candidate_meta[candidate_id]["actual_ep_ranks"])):
                errors.append(f"{candidate_id}: inconsistent topology metadata across roots")
            if meta.get("status") != "workload_passed":
                errors.append(f"{candidate_id}: {root} status={meta.get('status')}")
            stats = peak_gpu_stats(root, root_gpu_ids)
            root_stats.append(stats)
            root_audit.append({
                "candidate_id": candidate_id, "root": str(root),
                "meta_sha256": sha256(meta_path), "status": meta.get("status"), **stats,
            })
            for summary_path in sorted((root / "workloads").glob("*/summary.json")):
                cell_id = summary_path.parent.name
                if cell_id in summaries[candidate_id]:
                    errors.append(f"{candidate_id}: duplicate cell {cell_id}")
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["artifact_path"] = str(summary_path)
                summary["artifact_sha256"] = sha256(summary_path)
                summaries[candidate_id][cell_id] = summary
        if candidate_id in candidate_meta:
            candidate_meta[candidate_id].update({
                "peak_selected_gpu_memory_sum_mib": max(
                    (row["peak_selected_gpu_memory_sum_mib"] for row in root_stats
                     if row["peak_selected_gpu_memory_sum_mib"] is not None), default=None),
                "peak_selected_gpu_power_sum_w": max(
                    (row["peak_selected_gpu_power_sum_w"] for row in root_stats
                     if row["peak_selected_gpu_power_sum_w"] is not None), default=None),
            })

    rows = []
    chat_template_hashes: set[str] = set()
    for cell_id, cell in expected.items():
        measured: dict[str, Any] = {}
        for candidate_id in roots_by_candidate:
            summary = summaries[candidate_id].get(cell_id)
            if summary is None:
                errors.append(f"{candidate_id}: missing expected cell {cell_id}")
                continue
            if summary.get("failed") != 0:
                errors.append(f"{candidate_id}/{cell_id}: failed={summary.get('failed')}")
            if summary.get("completed") != summary.get("requests"):
                errors.append(
                    f"{candidate_id}/{cell_id}: completed={summary.get('completed')} "
                    f"requests={summary.get('requests')}")
            if summary.get("input_tokens_actual") != cell["input_tokens"]:
                errors.append(
                    f"{candidate_id}/{cell_id}: actual tokens "
                    f"{summary.get('input_tokens_actual')} != {cell['input_tokens']}")
            if summary.get("concurrency") != cell["concurrency"]:
                errors.append(f"{candidate_id}/{cell_id}: concurrency mismatch")
            template_hash = summary.get("chat_template_sha256")
            if template_hash:
                chat_template_hashes.add(template_hash)
            p99 = (summary.get("e2e_ms") or {}).get("p99")
            if not isinstance(p99, (int, float)):
                errors.append(f"{candidate_id}/{cell_id}: missing numeric e2e p99")
            wall = summary.get("wall_time_s")
            tokens = summary.get("output_tokens_total")
            throughput = (tokens / wall if isinstance(tokens, (int, float)) and
                          isinstance(wall, (int, float)) and wall > 0 else None)
            measured[candidate_id] = {
                "e2e_p50_ms": summary["e2e_ms"]["p50"],
                "e2e_p95_ms": summary["e2e_ms"]["p95"],
                "e2e_p99_ms": p99,
                "ttft_p99_ms": summary["ttft_ms"]["p99"],
                "tpot_p99_ms": summary["tpot_ms"]["p99"],
                "output_tokens_s": throughput,
                "completed": summary["completed"], "failed": summary["failed"],
                "artifact_path": summary["artifact_path"],
                "artifact_sha256": summary["artifact_sha256"],
            }
        winner = min(measured, key=lambda key: measured[key]["e2e_p99_ms"]) if measured else None
        rows.append({"workload_id": cell_id, **cell, "measurements": measured,
                     "oracle_by_e2e_p99": winner})

    if len(chat_template_hashes) != 1:
        errors.append(f"chat-template SHA count is {len(chat_template_hashes)}, expected 1")

    aggregate = []
    for candidate_id, meta in candidate_meta.items():
        candidate_rows = [row["measurements"][candidate_id] for row in rows
                          if candidate_id in row["measurements"]]
        aggregate.append({
            "candidate_id": candidate_id, **meta,
            "expected_cells_completed": len(candidate_rows),
            "oracle_wins": sum(row["oracle_by_e2e_p99"] == candidate_id for row in rows),
            "median_e2e_p99_ms": median([row["e2e_p99_ms"] for row in candidate_rows]),
            "median_tpot_p99_ms": median([row["tpot_p99_ms"] for row in candidate_rows]),
            "median_output_tokens_s": median([row["output_tokens_s"] for row in candidate_rows]),
        })
    pareto = [row["candidate_id"] for row in aggregate
              if not any(dominates(other, row) for other in aggregate if other is not row)]

    extras = {
        candidate_id: sorted(set(candidate_summaries) - set(expected))
        for candidate_id, candidate_summaries in summaries.items()
    }
    result = {
        "schema_version": "qtopomoe.phase8_single_pass_screen.v1",
        "status": "accepted" if not errors else "rejected",
        "gate": {
            "all_expected_cells_present": not any("missing expected cell" in item for item in errors),
            "failed_zero": not any("failed=" in item for item in errors),
            "token_exact": not any("actual tokens" in item for item in errors),
            "single_chat_template": len(chat_template_hashes) == 1,
            "ep_rank_truth": not any("EP rank" in item for item in errors),
            "all_requests_completed": not any("completed=" in item for item in errors),
        },
        "workload_matrix": str(args.workload_matrix),
        "workload_matrix_sha256": sha256(args.workload_matrix),
        "chat_template_sha256": next(iter(chat_template_hashes), None),
        "candidate_aggregate": aggregate,
        "resource_aware_pareto_candidates": pareto,
        "rows": rows,
        "extra_non_runbook_cells": extras,
        "root_audit": root_audit,
        "errors": errors,
        "interpretation": (
            "Single-pass screening selects candidates for later randomized five-repeat runs; "
            "it is not a confidence-interval result."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "pareto": pareto,
                      "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
