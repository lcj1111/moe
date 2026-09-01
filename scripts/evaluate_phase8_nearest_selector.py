#!/usr/bin/env python3
# 作用：用留族近邻基线评估 Phase 8 workload 选择效果。
"""用最简近邻策略对 Phase 8 正式矩阵做留族验证。

选择器先严格匹配 prefix-cache 与到达模式，再按输入、输出、并发和冻结
请求率的对数距离寻找一个已测 workload。测试 workload 所属的 W1/W2/W3/W4
族在选择时整体排除，避免把同族 oracle 标签泄漏到预测中。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def family(row: dict[str, Any]) -> str:
    return str(row["base_cell_id"]).split("_", 1)[0]


def measured(row: dict[str, Any], candidate: str) -> float:
    return float(row["measurements"][candidate]["metrics"]["e2e_p99_ms"]["median"])


def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    total = sum(
        math.log2(float(left[key]) / float(right[key])) ** 2
        for key in ("input_tokens", "output_tokens", "concurrency")
    )
    left_rate = left.get("request_rate_rps")
    right_rate = right.get("request_rate_rps")
    if left_rate is not None and right_rate is not None:
        total += math.log2(float(left_rate) / float(right_rate)) ** 2
    return total


def choose(
    row: dict[str, Any], training: list[dict[str, Any]], candidates: list[str]
) -> tuple[str, dict[str, Any]]:
    eligible = [
        candidate_row for candidate_row in training
        if candidate_row["prefix_cache_pct"] == row["prefix_cache_pct"]
        and candidate_row["arrival_mode"] == row["arrival_mode"]
    ]
    if not eligible:
        raise ValueError(f"no control-matched training row for {row['workload_id']}")
    nearest = min(
        eligible,
        key=lambda candidate_row: (
            distance(row, candidate_row), candidate_row["workload_id"]
        ),
    )
    selected = min(
        candidates, key=lambda candidate: (measured(nearest, candidate), candidate)
    )
    return selected, nearest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overhead-repeats", type=int, default=1000)
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if aggregate.get("status") != "accepted":
        raise SystemExit("formal aggregate Gate is not accepted")
    rows = aggregate["rows"]
    candidates = sorted(rows[0]["measurements"])
    expected = set(candidates)
    if any(set(row["measurements"]) != expected for row in rows):
        raise SystemExit("candidate coverage differs across workload cells")

    evaluation: list[dict[str, Any]] = []
    overhead_ms: list[float] = []
    folds = sorted({family(row) for row in rows})
    for held_out in folds:
        training = [row for row in rows if family(row) != held_out]
        test = [row for row in rows if family(row) == held_out]
        for row in test:
            started = time.perf_counter()
            selected = nearest = None
            for _ in range(args.overhead_repeats):
                selected, nearest = choose(row, training, candidates)
            elapsed = (time.perf_counter() - started) * 1000 / args.overhead_repeats
            overhead_ms.append(elapsed)
            values = {candidate: measured(row, candidate) for candidate in candidates}
            oracle = min(candidates, key=lambda candidate: (values[candidate], candidate))
            regret = (values[selected] - values[oracle]) / values[oracle] * 100
            oracle_ci = row["measurements"][oracle]["metrics"]["e2e_p99_ms"]
            other_ci_lows = [
                float(row["measurements"][candidate]["metrics"]["e2e_p99_ms"]
                      ["bootstrap_95ci_low"])
                for candidate in candidates if candidate != oracle
            ]
            evaluation.append({
                "workload_id": row["workload_id"],
                "held_out_family": held_out,
                "selected_candidate": selected,
                "oracle_candidate": oracle,
                "selected_p99_ms": values[selected],
                "oracle_p99_ms": values[oracle],
                "regret_pct": regret,
                "nearest_training_workload_id": nearest["workload_id"],
                "nearest_distance": distance(row, nearest),
                "decision_overhead_ms": elapsed,
                "decision_overhead_pct_of_service_p99": elapsed / values[selected] * 100,
                "oracle_bootstrap_separated": (
                    float(oracle_ci["bootstrap_95ci_high"]) < min(other_ci_lows)
                ),
                "infeasible_selection": selected not in row["measurements"],
            })

    regrets = [float(row["regret_pct"]) for row in evaluation]
    overhead_pct = [
        float(row["decision_overhead_pct_of_service_p99"]) for row in evaluation
    ]
    infeasible_rate = sum(row["infeasible_selection"] for row in evaluation) / len(evaluation)
    metrics = {
        "top1_accuracy": sum(value == 0 for value in regrets) / len(regrets),
        "median_regret_pct": statistics.median(regrets),
        "p95_regret_pct": percentile(regrets, 0.95),
        "decision_overhead_ms_p95": percentile(overhead_ms, 0.95),
        "decision_overhead_pct_p95": percentile(overhead_pct, 0.95),
        "infeasible_misselection_rate": infeasible_rate,
        "oracle_bootstrap_separated_cells": sum(
            row["oracle_bootstrap_separated"] for row in evaluation
        ),
        "selected_candidate_counts": dict(Counter(
            row["selected_candidate"] for row in evaluation
        )),
        "oracle_candidate_counts": dict(Counter(
            row["oracle_candidate"] for row in evaluation
        )),
    }
    gates = {
        "median_regret_le_5pct": metrics["median_regret_pct"] <= 5,
        "p95_regret_le_10pct": metrics["p95_regret_pct"] <= 10,
        "controller_overhead_lt_1pct": metrics["decision_overhead_pct_p95"] < 1,
        "infeasible_misselection_zero": infeasible_rate == 0,
    }
    result = {
        "schema_version": "qtopomoe.phase8_nearest_selector_gate.v1",
        "status": "accepted" if all(gates.values()) else "gate_failed",
        "method": {
            "selector": "single_nearest_measured_workload",
            "control_match": ["prefix_cache_pct", "arrival_mode"],
            "distance": (
                "sum of squared log2 ratios for input_tokens, output_tokens, "
                "concurrency and request_rate_rps when present"
            ),
            "validation": "leave_one_workload_family_out_W1_W2_W3_W4",
            "leakage_control": "the complete held-out family is excluded from lookup",
            "overhead_repeats_per_cell": args.overhead_repeats,
        },
        "input": {
            "aggregate": str(args.aggregate),
            "aggregate_sha256": sha256(args.aggregate),
            "aggregate_schema_version": aggregate.get("schema_version"),
        },
        "candidate_count": len(candidates),
        "workload_count": len(rows),
        "folds": folds,
        "metrics": metrics,
        "gates": gates,
        "formal_gate_ready": all(gates.values()),
        "rows": evaluation,
        "conclusion": (
            "简化选择器用于替代旧模型进行误差分析；所有 Gate 全部通过前，"
            "不得在在线服务中启用。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
