#!/usr/bin/env python3
"""在互不重叠的训练集与测试集上执行 Phase 8 selector 正式 Gate。"""
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
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def nested(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{row.get('workload_id')}: 缺少 selector 特征 {path}")
        value = value[key]
    return value


def workload_fingerprint(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in (
        "input_tokens", "output_tokens", "concurrency", "prefix_cache_pct",
        "arrival_mode", "request_rate_rps",
    ))


def validate_pre_decision_state(row: dict[str, Any], require_server_queue: bool) -> None:
    state = row.get("selector_state")
    if not isinstance(state, dict):
        raise ValueError(f"{row.get('workload_id')}: 缺少独立 selector_state")
    if state.get("observation_phase") != "pre_decision" or not state.get(
            "decision_eligible"):
        raise ValueError(f"{row.get('workload_id')}: selector_state 不是决策前窗口")
    if require_server_queue and not state.get("server_queue_depth_available"):
        raise ValueError(f"{row.get('workload_id')}: 缺少真实服务端队列遥测")


def transformed(value: Any, spec: dict[str, Any]) -> float:
    number = float(value)
    transform = spec.get("transform", "linear")
    if transform == "log2":
        if number <= 0:
            raise ValueError(f"{spec['path']} 必须为正数")
        return math.log2(number)
    if transform == "log1p":
        if number < 0:
            raise ValueError(f"{spec['path']} 不能为负数")
        return math.log1p(number)
    if transform == "linear":
        return number / float(spec.get("scale", 1.0))
    raise ValueError(f"未知特征变换：{transform}")


def distance(left: dict[str, Any], right: dict[str, Any],
             features: list[dict[str, Any]]) -> float:
    total = 0.0
    for spec in features:
        left_value = nested(left, spec["path"])
        right_value = nested(right, spec["path"])
        if left_value is None or right_value is None:
            if (left_value is None and right_value is None
                    and spec.get("allow_both_null") is True):
                continue
            if ((left_value is None) != (right_value is None)
                    and "missing_mismatch_penalty" in spec):
                penalty = float(spec["missing_mismatch_penalty"])
                if not math.isfinite(penalty) or penalty < 0:
                    raise ValueError(
                        f"{spec['path']}: missing_mismatch_penalty 必须是有限非负数")
                total += float(spec.get("weight", 1.0)) * penalty
                continue
            raise ValueError(f"{spec['path']}: selector 特征只有一侧缺失")
        total += float(spec.get("weight", 1.0)) * (
            transformed(left_value, spec) - transformed(right_value, spec)
        ) ** 2
    return total


def measured(row: dict[str, Any], candidate: str) -> float:
    return float(row["measurements"][candidate]["metrics"]["e2e_p99_ms"]["median"])


def choose(row: dict[str, Any], training: list[dict[str, Any]],
           candidates: list[str], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    strict = config["strict_match"]
    eligible = [candidate_row for candidate_row in training if all(
        nested(candidate_row, path) == nested(row, path) for path in strict)]
    if not eligible:
        raise ValueError(f"{row['workload_id']}: 没有满足严格条件的训练样本")
    ranked = sorted((
        (distance(row, candidate_row, config["numeric_features"]), candidate_row)
        for candidate_row in eligible
    ), key=lambda item: (item[0], item[1]["workload_id"]))
    nearest = ranked[0][1]
    selector = config.get("selector", "telemetry_aware_single_nearest_workload")
    if selector == "telemetry_aware_single_nearest_workload":
        selected = min(candidates, key=lambda candidate: (
            measured(nearest, candidate), candidate))
        return selected, nearest
    if selector != "telemetry_aware_knn_cost":
        raise ValueError(f"未知 selector：{selector}")
    neighbor_count = int(config.get("neighbor_count", 3))
    if neighbor_count <= 0:
        raise ValueError("neighbor_count 必须为正整数")
    neighbors = ranked[:min(neighbor_count, len(ranked))]
    weighting = config.get("neighbor_weighting", "inverse_distance")
    normalize = bool(config.get("normalize_training_cost_by_oracle", True))
    epsilon = float(config.get("distance_epsilon", 1e-9))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("distance_epsilon 必须为有限正数")

    def neighbor_weight(value: float) -> float:
        if weighting == "uniform":
            return 1.0
        if weighting == "inverse_distance":
            return 1.0 / (math.sqrt(value) + epsilon)
        raise ValueError(f"未知邻居权重：{weighting}")

    predicted_costs: dict[str, float] = {}
    for candidate in candidates:
        weighted_sum = weight_sum = 0.0
        for neighbor_distance, neighbor in neighbors:
            value = measured(neighbor, candidate)
            if normalize:
                value /= min(measured(neighbor, item) for item in candidates)
            weight = neighbor_weight(neighbor_distance)
            weighted_sum += weight * value
            weight_sum += weight
        predicted_costs[candidate] = weighted_sum / weight_sum
    selected = min(candidates, key=lambda candidate: (
        predicted_costs[candidate], candidate))
    return selected, nearest


def evaluate(training: dict[str, Any], test: dict[str, Any],
             config: dict[str, Any], overhead_repeats: int) -> dict[str, Any]:
    if training.get("status") != "accepted" or test.get("status") != "accepted":
        raise ValueError("训练或测试聚合 Gate 未接受")
    if config.get("status") != "frozen":
        raise ValueError("selector 配置未冻结，禁止读取独立测试集")
    training_rows, test_rows = training["rows"], test["rows"]
    train_ids = {row["workload_id"] for row in training_rows}
    test_ids = {row["workload_id"] for row in test_rows}
    if train_ids & test_ids:
        raise ValueError("训练集与测试集 workload_id 重叠")
    train_fingerprints = {workload_fingerprint(row) for row in training_rows}
    test_fingerprints = {workload_fingerprint(row) for row in test_rows}
    if train_fingerprints & test_fingerprints:
        raise ValueError("训练集与测试集 workload 参数指纹重叠")
    require_server_queue = bool(config.get("require_server_queue_telemetry", True))
    for row in [*training_rows, *test_rows]:
        validate_pre_decision_state(row, require_server_queue)
    candidates = sorted(training_rows[0]["measurements"])
    expected = set(candidates)
    for row in [*training_rows, *test_rows]:
        if set(row["measurements"]) != expected:
            raise ValueError(f"{row['workload_id']}: 候选覆盖不一致")

    rows: list[dict[str, Any]] = []
    overhead: list[float] = []
    for row in test_rows:
        selected = nearest = None
        started = time.perf_counter()
        for _ in range(overhead_repeats):
            selected, nearest = choose(row, training_rows, candidates, config)
        elapsed_ms = (time.perf_counter() - started) * 1000 / overhead_repeats
        overhead.append(elapsed_ms)
        values = {candidate: measured(row, candidate) for candidate in candidates}
        oracle = min(candidates, key=lambda candidate: (values[candidate], candidate))
        regret = (values[selected] - values[oracle]) / values[oracle] * 100
        rows.append({
            "workload_id": row["workload_id"],
            "selected_candidate": selected,
            "oracle_candidate": oracle,
            "selected_p99_ms": values[selected],
            "oracle_p99_ms": values[oracle],
            "regret_pct": regret,
            "nearest_training_workload_id": nearest["workload_id"],
            "nearest_distance": distance(row, nearest, config["numeric_features"]),
            "decision_overhead_ms": elapsed_ms,
            "decision_overhead_pct_of_service_p99": elapsed_ms / values[selected] * 100,
            "infeasible_selection": selected not in row["measurements"],
        })
    regrets = [float(row["regret_pct"]) for row in rows]
    overhead_pct = [float(row["decision_overhead_pct_of_service_p99"]) for row in rows]
    infeasible_rate = sum(row["infeasible_selection"] for row in rows) / len(rows)
    metrics = {
        "top1_accuracy": sum(value == 0 for value in regrets) / len(regrets),
        "median_regret_pct": statistics.median(regrets),
        "p95_regret_pct": percentile(regrets, 0.95),
        "decision_overhead_ms_p95": percentile(overhead, 0.95),
        "decision_overhead_pct_p95": percentile(overhead_pct, 0.95),
        "infeasible_misselection_rate": infeasible_rate,
        "selected_candidate_counts": dict(Counter(
            row["selected_candidate"] for row in rows)),
        "oracle_candidate_counts": dict(Counter(
            row["oracle_candidate"] for row in rows)),
    }
    thresholds = config["gates"]
    gates = {
        "median_regret": metrics["median_regret_pct"] <= thresholds[
            "median_regret_pct_max"],
        "p95_regret": metrics["p95_regret_pct"] <= thresholds[
            "p95_regret_pct_max"],
        "controller_overhead": metrics["decision_overhead_pct_p95"] < thresholds[
            "decision_overhead_pct_max"],
        "infeasible_misselection": infeasible_rate <= thresholds[
            "infeasible_misselection_rate_max"],
    }
    return {"rows": rows, "metrics": metrics, "gates": gates,
            "status": "accepted" if all(gates.values()) else "gate_failed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-aggregate", type=Path, required=True)
    parser.add_argument("--test-aggregate", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overhead-repeats", type=int, default=1000)
    args = parser.parse_args()
    training = json.loads(args.training_aggregate.read_text(encoding="utf-8"))
    test = json.loads(args.test_aggregate.read_text(encoding="utf-8"))
    config = json.loads(args.selector_config.read_text(encoding="utf-8"))
    evaluated = evaluate(training, test, config, args.overhead_repeats)
    result = {
        "schema_version": "qtopomoe.phase8_independent_selector_gate.v1",
        **evaluated,
        "method": {
            "validation": "frozen_selector_on_disjoint_independent_workloads",
            "leakage_control": (
                "训练/测试 ID 与参数指纹均不重叠；只接受 pre_decision 状态；"
                "读取测试集前 selector 配置必须为 frozen。"
            ),
        },
        "inputs": {
            "training_aggregate": str(args.training_aggregate),
            "training_aggregate_sha256": sha256(args.training_aggregate),
            "test_aggregate": str(args.test_aggregate),
            "test_aggregate_sha256": sha256(args.test_aggregate),
            "selector_config": str(args.selector_config),
            "selector_config_sha256": sha256(args.selector_config),
        },
        "formal_gate_ready": evaluated["status"] == "accepted",
        "conclusion": (
            "只有本独立 Gate 全部通过后，selector 才能进入动态 trigger、cooldown、"
            "rollback 在线闭环；旧的同矩阵交叉验证不能替代本 Gate。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["metrics"]},
                     ensure_ascii=False, sort_keys=True))
    if result["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
