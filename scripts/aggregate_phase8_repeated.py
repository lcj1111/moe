#!/usr/bin/env python3
"""Gate and bootstrap the randomized five-repeat Phase 8 Pareto run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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


def stable_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def bootstrap_median(values: list[float], seed: int,
                     samples: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(values, k=len(values)))
                   for _ in range(samples))
    low = draws[int(0.025 * (samples - 1))]
    high = draws[int(0.975 * (samples - 1))]
    return {"n": len(values), "median": statistics.median(values),
            "bootstrap_95ci_low": low, "bootstrap_95ci_high": high,
            "bootstrap_samples": samples}


def peak_memory(root: Path, gpu_ids: set[int]) -> float | None:
    path = root / "gpu_memory.csv"
    if not path.exists():
        return None
    samples: list[float] = []
    seen: set[int] = set()
    total = 0.0
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5 or row[0].strip() == "timestamp":
                continue
            try:
                index = int(row[1].strip())
                if index not in gpu_ids:
                    continue
                if index in seen:
                    samples.append(total)
                    seen, total = set(), 0.0
                seen.add(index)
                total += float(row[2].strip())
            except ValueError:
                continue
    if seen:
        samples.append(total)
    return max(samples, default=None)


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (left["median_of_cell_median_e2e_p99_ms"],
                   -left["median_of_cell_median_output_tokens_s"], left["gpu_count"])
    right_values = (right["median_of_cell_median_e2e_p99_ms"],
                    -right["median_of_cell_median_output_tokens_s"], right["gpu_count"])
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    schedule_path = args.run_root / "schedule.json"
    schedule = load_json(schedule_path)
    matrix_path = Path(schedule["workload_matrix"])
    matrix = load_json(matrix_path)
    expected_cells = {row["id"]: row for row in matrix["cells"]}
    errors: list[str] = []
    records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    candidate_topology: dict[str, dict[str, Any]] = {}
    run_audit: list[dict[str, Any]] = []
    template_hashes: set[str] = set()

    for item in schedule["schedule"]:
        run_id = f"{item['candidate_id']}__r{item['repeat']:02d}"
        run_dir = args.run_root / "runs" / run_id
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            errors.append(f"missing run {run_id}")
            continue
        meta = load_json(meta_path)
        if meta.get("status") != "workload_passed":
            errors.append(f"{run_id}: status={meta.get('status')}")
        if meta.get("actual_ep_ranks") != meta.get("expected_ep_ranks"):
            errors.append(f"{run_id}: EP rank mismatch")
        topology = {
            "gpu_ids": meta["gpu_ids"], "gpu_count": len(meta["gpu_ids"]),
            "tp": meta["tp"], "dp": meta["dp"],
            "ep_enabled": meta["ep_enabled"],
            "expected_ep_ranks": meta["expected_ep_ranks"],
            "actual_ep_ranks": meta["actual_ep_ranks"],
            "model_path": meta["model_path"],
        }
        previous = candidate_topology.setdefault(item["candidate_id"], topology)
        if previous != topology:
            errors.append(f"{item['candidate_id']}: topology changed across repeats")
        memory = peak_memory(run_dir, set(meta["gpu_ids"]))
        run_audit.append({
            "run_id": run_id, "meta_sha256": sha256(meta_path),
            "status": meta.get("status"),
            "peak_selected_gpu_memory_sum_mib": memory,
        })
        for cell_id, cell in expected_cells.items():
            summary_path = run_dir / "workloads" / cell_id / "summary.json"
            if not summary_path.exists():
                errors.append(f"{run_id}: missing cell {cell_id}")
                continue
            summary = load_json(summary_path)
            if summary.get("failed") != 0:
                errors.append(f"{run_id}/{cell_id}: failed={summary.get('failed')}")
            if summary.get("completed") != summary.get("requests"):
                errors.append(f"{run_id}/{cell_id}: incomplete requests")
            if summary.get("input_tokens_actual") != cell["input_tokens"]:
                errors.append(f"{run_id}/{cell_id}: input-token mismatch")
            if summary.get("concurrency") != cell["concurrency"]:
                errors.append(f"{run_id}/{cell_id}: concurrency mismatch")
            if summary.get("chat_template_sha256"):
                template_hashes.add(summary["chat_template_sha256"])
            wall = summary.get("wall_time_s")
            tokens = summary.get("output_tokens_total")
            throughput = tokens / wall if isinstance(wall, (int, float)) and wall > 0 else None
            records[item["candidate_id"]][cell_id].append({
                "repeat": item["repeat"], "e2e_p99_ms": summary["e2e_ms"]["p99"],
                "ttft_p99_ms": summary["ttft_ms"]["p99"],
                "tpot_p99_ms": summary["tpot_ms"]["p99"],
                "output_tokens_s": throughput, "artifact_path": str(summary_path),
                "artifact_sha256": sha256(summary_path),
            })

    if len(template_hashes) != 1:
        errors.append(f"chat-template SHA count={len(template_hashes)}, expected 1")
    expected_repeats = len(schedule["schedule"]) // len(candidate_topology) if candidate_topology else 0
    rows = []
    for cell_id, cell in expected_cells.items():
        measurements = {}
        for candidate_id in candidate_topology:
            values = records[candidate_id][cell_id]
            if len(values) != expected_repeats:
                errors.append(
                    f"{candidate_id}/{cell_id}: repeats={len(values)} expected={expected_repeats}")
            metrics = {}
            for metric in ("e2e_p99_ms", "ttft_p99_ms", "tpot_p99_ms",
                           "output_tokens_s"):
                metric_values = [float(row[metric]) for row in values
                                 if isinstance(row.get(metric), (int, float))]
                if metric_values:
                    metrics[metric] = bootstrap_median(
                        metric_values,
                        stable_seed(f"{candidate_id}/{cell_id}/{metric}", schedule["seed"]),
                        args.bootstrap_samples)
            measurements[candidate_id] = {"metrics": metrics, "repeats": values}
        oracle = min(measurements, key=lambda key: measurements[key]["metrics"]
                     ["e2e_p99_ms"]["median"]) if measurements else None
        rows.append({"workload_id": cell_id, **cell, "measurements": measurements,
                     "oracle_by_median_e2e_p99": oracle})

    aggregate = []
    for candidate_id, topology in candidate_topology.items():
        candidate_rows = [row["measurements"][candidate_id]["metrics"] for row in rows]
        memory_values = [row["peak_selected_gpu_memory_sum_mib"] for row in run_audit
                         if row["run_id"].startswith(candidate_id + "__") and
                         row["peak_selected_gpu_memory_sum_mib"] is not None]
        aggregate.append({
            "candidate_id": candidate_id, **topology,
            "median_of_cell_median_e2e_p99_ms": statistics.median(
                row["e2e_p99_ms"]["median"] for row in candidate_rows),
            "median_of_cell_median_output_tokens_s": statistics.median(
                row["output_tokens_s"]["median"] for row in candidate_rows),
            "oracle_wins": sum(row["oracle_by_median_e2e_p99"] == candidate_id
                               for row in rows),
            "peak_selected_gpu_memory_sum_mib": max(memory_values, default=None),
        })
    pareto = [row["candidate_id"] for row in aggregate
              if not any(dominates(other, row) for other in aggregate if other is not row)]
    result = {
        "schema_version": "qtopomoe.phase8_repeated_bootstrap.v1",
        "status": "accepted" if not errors else "rejected",
        "gate": {
            "all_runs_passed": not any("status=" in error for error in errors),
            "all_cells_present": not any("missing cell" in error for error in errors),
            "five_repeats": not any("repeats=" in error for error in errors),
            "failed_zero": not any("failed=" in error for error in errors),
            "request_complete": not any("incomplete requests" in error for error in errors),
            "token_exact": not any("input-token mismatch" in error for error in errors),
            "ep_rank_truth": not any("EP rank" in error for error in errors),
            "single_chat_template": len(template_hashes) == 1,
        },
        "schedule": str(schedule_path), "schedule_sha256": sha256(schedule_path),
        "workload_matrix": str(matrix_path),
        "workload_matrix_sha256": sha256(matrix_path),
        "chat_template_sha256": next(iter(template_hashes), None),
        "bootstrap_samples": args.bootstrap_samples,
        "candidate_aggregate": aggregate,
        "resource_aware_pareto_candidates": pareto,
        "rows": rows, "run_audit": run_audit, "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "pareto": pareto,
                      "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
