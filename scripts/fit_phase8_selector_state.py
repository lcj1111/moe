#!/usr/bin/env python3
"""只使用训练 outcome 与决策前状态拟合并冻结透明 Phase 8 selector。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_phase8_independent_selector import (
        choose, measured, percentile, validate_pre_decision_state,
    )
except ModuleNotFoundError:  # 兼容 ``python scripts/fit_*.py`` 直接执行
    from evaluate_phase8_independent_selector import (  # type: ignore[no-redef]
        choose, measured, percentile, validate_pre_decision_state,
    )


GROUPS = {
    "shape": ("input_tokens", "output_tokens", "concurrency"),
    "cache": ("actual_cache_hit_ratio",),
    "arrival": ("recent_arrival_rate_rps",),
    "queue": ("client_queue_delay", "server_queue_waiting",
              "server_requests_running"),
    "kv": ("server_kv_cache_usage",),
    "short_p99": ("short_window",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family(row: dict[str, Any]) -> str:
    value = str(row.get("base_cell_id") or row["workload_id"])
    return value.split("_", 1)[0].lower()


def feature_group(path: str) -> str:
    for group, fragments in GROUPS.items():
        if any(fragment in path for fragment in fragments):
            return group
    raise ValueError(f"没有为特征定义权重组：{path}")


def weighted_config(template: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    result = json.loads(json.dumps(template))
    for feature in result["numeric_features"]:
        group = feature_group(feature["path"])
        feature["weight"] = float(feature.get("weight", 1.0)) * weights[group]
    return result


def evaluate_cv(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    candidates = sorted(rows[0]["measurements"])
    evaluation = []
    folds = sorted({family(row) for row in rows})
    for held_out in folds:
        training = [row for row in rows if family(row) != held_out]
        test = [row for row in rows if family(row) == held_out]
        if not training or not test:
            raise ValueError(f"无效训练折：{held_out}")
        for row in test:
            selected, nearest = choose(row, training, candidates, config)
            values = {candidate: measured(row, candidate) for candidate in candidates}
            oracle = min(candidates, key=lambda candidate: (values[candidate], candidate))
            regret = (values[selected] - values[oracle]) / values[oracle] * 100
            evaluation.append({
                "workload_id": row["workload_id"],
                "held_out_family": held_out,
                "selected_candidate": selected,
                "oracle_candidate": oracle,
                "nearest_training_workload_id": nearest["workload_id"],
                "regret_pct": regret,
            })
    regrets = [row["regret_pct"] for row in evaluation]
    return {
        "folds": folds,
        "rows": evaluation,
        "metrics": {
            "top1_accuracy": sum(value == 0 for value in regrets) / len(regrets),
            "median_regret_pct": statistics.median(regrets),
            "p95_regret_pct": percentile(regrets, 0.95),
            "maximum_regret_pct": max(regrets),
        },
    }


def fit(aggregate: dict[str, Any], template: dict[str, Any],
        grid: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    if aggregate.get("status") != "accepted":
        raise ValueError("训练 aggregate Gate 未接受")
    if template.get("status") not in ("draft_not_fitted", "training_gate_failed"):
        raise ValueError("输入必须是未拟合模板，禁止覆盖已经冻结的配置")
    rows = aggregate["rows"]
    for row in rows:
        validate_pre_decision_state(
            row, bool(template.get("require_server_queue_telemetry", True)))
    candidates = set(rows[0]["measurements"])
    if any(set(row["measurements"]) != candidates for row in rows):
        raise ValueError("训练矩阵候选覆盖不一致")

    tunable = ("cache", "arrival", "queue", "kv", "short_p99")
    trials = []
    for values in itertools.product(grid, repeat=len(tunable)):
        weights = {"shape": 1.0, **dict(zip(tunable, values))}
        config = weighted_config(template, weights)
        try:
            cv = evaluate_cv(rows, config)
        except ValueError:
            continue
        metrics = cv["metrics"]
        trials.append({"weights": weights, "metrics": metrics})
    if not trials:
        raise ValueError("没有可评估的权重组合")
    best = min(trials, key=lambda row: (
        row["metrics"]["p95_regret_pct"],
        row["metrics"]["median_regret_pct"],
        -row["metrics"]["top1_accuracy"],
        tuple(row["weights"][key] for key in tunable),
    ))
    fitted = weighted_config(template, best["weights"])
    thresholds = fitted["gates"]
    training_gates = {
        "median_regret": best["metrics"]["median_regret_pct"]
                         <= thresholds["median_regret_pct_max"],
        "p95_regret": best["metrics"]["p95_regret_pct"]
                      <= thresholds["p95_regret_pct_max"],
    }
    fitted.update({
        "status": "frozen" if all(training_gates.values()) else "training_gate_failed",
        "fit_method": "leave_one_W_family_out_group_weight_grid_search",
        "fit_weights": best["weights"],
        "training_metrics": best["metrics"],
        "training_gates": training_gates,
        "independent_test_read": False,
    })
    report = {
        "schema_version": "qtopomoe.phase8_selector_fit_report.v1",
        "status": fitted["status"],
        "grid": grid,
        "trial_count": len(trials),
        "best": best,
        "training_gates": training_gates,
        "top_trials": sorted(trials, key=lambda row: (
            row["metrics"]["p95_regret_pct"],
            row["metrics"]["median_regret_pct"],
            -row["metrics"]["top1_accuracy"],
        ))[:20],
    }
    return fitted, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-aggregate", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--weight-grid", default="0.25,1,4")
    args = parser.parse_args()
    aggregate = json.loads(args.training_aggregate.read_text(encoding="utf-8"))
    template = json.loads(args.template.read_text(encoding="utf-8"))
    grid = [float(value) for value in args.weight_grid.split(",")]
    if any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("权重网格必须是有限正数")
    fitted, report = fit(aggregate, template, grid)
    source = {
        "training_aggregate": str(args.training_aggregate),
        "training_aggregate_sha256": sha256(args.training_aggregate),
        "template": str(args.template),
        "template_sha256": sha256(args.template),
    }
    fitted["fit_sources"] = source
    report["fit_sources"] = source
    for path, value in ((args.output_config, fitted), (args.output_report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": fitted["status"], **report["best"]["metrics"]},
                     ensure_ascii=False, sort_keys=True))
    if fitted["status"] != "frozen":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
