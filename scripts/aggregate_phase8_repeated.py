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


def controlled_summary_errors(summary: dict[str, Any], cell: dict[str, Any],
                              label: str) -> list[str]:
    """Return formal cache/arrival contract violations for a controlled cell."""
    if "prefix_cache_pct" not in cell and "arrival_mode" not in cell:
        return []
    errors = []
    prefix = summary.get("prefix_cache", {})
    arrival = summary.get("arrival", {})
    expected_mode = cell.get("arrival_mode", "closed_loop")
    if summary.get("server_prompt_tokens_exact") is not True:
        errors.append(f"{label}: server prompt-token Gate failed")
    if prefix.get("target_pct") != cell.get("prefix_cache_pct", 0):
        errors.append(f"{label}: prefix target Gate failed")
    if prefix.get("usage_details_complete") is not True:
        errors.append(f"{label}: cache usage-detail Gate failed")
    if prefix.get("ratio_gate") is not True:
        errors.append(f"{label}: cache ratio Gate failed")
    if arrival.get("mode") != expected_mode:
        errors.append(f"{label}: arrival mode Gate failed")
    if arrival.get("schedule_gate") is not True:
        errors.append(f"{label}: arrival schedule Gate failed")
    if expected_mode == "closed_loop":
        if arrival.get("request_rate_target_rps") is not None:
            errors.append(f"{label}: closed-loop rate must be absent")
    else:
        actual_rate = arrival.get("request_rate_target_rps")
        expected_rate = cell.get("request_rate_rps")
        if (not isinstance(actual_rate, (int, float)) or
                not isinstance(expected_rate, (int, float)) or
                abs(float(actual_rate) - float(expected_rate)) > 1e-9):
            errors.append(f"{label}: frozen request-rate Gate failed")
    return errors


def latency_metric(summary: dict[str, Any], cell: dict[str, Any], metric: str) -> Any:
    """Use scheduled-arrival latency for controlled cells and legacy service latency otherwise."""
    controlled = "prefix_cache_pct" in cell or "arrival_mode" in cell
    if controlled and metric in ("e2e_ms", "ttft_ms"):
        offered = summary.get("offered_" + metric)
        if offered is not None:
            return offered
    return summary[metric]


def repeat_telemetry(summary: dict[str, Any]) -> dict[str, Any]:
    """Preserve selector-relevant telemetry without changing legacy Gates."""
    arrival = summary.get("arrival", {})
    service = summary.get("service_telemetry", {})
    return {
        "cache_expected_ratio": summary.get("prefix_cache", {}).get(
            "expected_cached_token_ratio"),
        "cache_actual_ratio": summary.get("prefix_cache", {}).get(
            "actual_cached_token_ratio"),
        "arrival_mode": arrival.get("mode"),
        "request_rate_target_rps": arrival.get("request_rate_target_rps"),
        "request_rate_realized_rps": arrival.get("request_rate_realized_rps"),
        "service_start_rate_realized_rps": arrival.get(
            "service_start_rate_realized_rps"),
        "arrival_lag_p95_s": arrival.get("scheduling_lag_s", {}).get("p95"),
        "client_queue_delay_p95_s": arrival.get("queue_delay_s", {}).get("p95"),
        "service_start_lag_p95_s": arrival.get(
            "service_start_lag_s", {}).get("p95"),
        "client_peak_in_flight": arrival.get("peak_in_flight"),
        "server_queue_waiting_p95": service.get(
            "num_requests_waiting", {}).get("p95"),
        "server_requests_running_p95": service.get(
            "num_requests_running", {}).get("p95"),
        "server_kv_cache_usage_p95": service.get(
            "kv_cache_usage_perc", {}).get("p95"),
        "server_telemetry_coverage_ratio": service.get("coverage_ratio"),
        "selector_state": summary.get("selector_state"),
    }


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
            errors.extend(controlled_summary_errors(
                summary, cell, f"{run_id}/{cell_id}"))
            if summary.get("chat_template_sha256"):
                template_hashes.add(summary["chat_template_sha256"])
            wall = summary.get("wall_time_s")
            tokens = summary.get("output_tokens_total")
            throughput = tokens / wall if isinstance(wall, (int, float)) and wall > 0 else None
            selected_e2e = latency_metric(summary, cell, "e2e_ms")
            selected_ttft = latency_metric(summary, cell, "ttft_ms")
            records[item["candidate_id"]][cell_id].append({
                "repeat": item["repeat"], "e2e_p99_ms": selected_e2e["p99"],
                "ttft_p99_ms": selected_ttft["p99"],
                "tpot_p99_ms": summary["tpot_ms"]["p99"],
                "output_tokens_s": throughput, "artifact_path": str(summary_path),
                "artifact_sha256": sha256(summary_path),
                "latency_clock": ("scheduled_arrival" if
                                  ("prefix_cache_pct" in cell or "arrival_mode" in cell)
                                  else "service_start"),
                "service_e2e_p99_ms": summary["e2e_ms"]["p99"],
                "service_ttft_p99_ms": summary["ttft_ms"]["p99"],
                **repeat_telemetry(summary),
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
            "server_prompt_tokens_exact": not any(
                "server prompt-token Gate" in error for error in errors),
            "controlled_cache": not any(
                ("cache " in error or "prefix target" in error) for error in errors),
            "controlled_arrival": not any(
                ("arrival " in error or "request-rate" in error or
                 "closed-loop rate" in error) for error in errors),
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
