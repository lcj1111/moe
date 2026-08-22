#!/usr/bin/env python3
"""只使用训练 outcome 与决策前状态拟合并冻结透明 Phase 8 selector。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_phase8_independent_selector import (
        choose, choose_regime_candidate, measured, percentile,
        validate_pre_decision_state,
    )
except ModuleNotFoundError:  # 兼容 ``python scripts/fit_*.py`` 直接执行
    from evaluate_phase8_independent_selector import (  # type: ignore[no-redef]
        choose, choose_regime_candidate, measured, percentile,
        validate_pre_decision_state,
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
            matched_rule_id = None
            if config.get("selector") == "telemetry_aware_regime_rules":
                _, matched_rule_id = choose_regime_candidate(row, candidates, config)
            evaluation.append({
                "workload_id": row["workload_id"],
                "held_out_family": held_out,
                "selected_candidate": selected,
                "oracle_candidate": oracle,
                "nearest_training_workload_id": nearest["workload_id"],
                "matched_rule_id": matched_rule_id,
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
        grid: list[float], neighbor_grid: list[int] | None = None,
        search_profile: str = "standard"
        ) -> tuple[dict[str, Any], dict[str, Any]]:
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
    neighbor_grid = neighbor_grid or [1, 3, 5, 7]
    standard_models = [{
        "selector": "telemetry_aware_single_nearest_workload",
        "neighbor_count": 1,
        "neighbor_weighting": "nearest",
        "normalize_training_cost_by_oracle": False,
    }]
    standard_models.extend({
        "selector": "telemetry_aware_knn_cost",
        "neighbor_count": neighbors,
        "neighbor_weighting": weighting,
        "normalize_training_cost_by_oracle": normalize,
    } for neighbors in neighbor_grid if neighbors > 1
      for weighting in ("uniform", "inverse_distance")
      for normalize in (False, True))
    trial_inputs: list[tuple[dict[str, float], dict[str, Any]]] = []
    search_space: dict[str, Any]
    if search_profile == "standard":
        for model in standard_models:
            for values in itertools.product(grid, repeat=len(tunable)):
                trial_inputs.append((
                    {"shape": 1.0, **dict(zip(tunable, values))}, model))
        search_space = {
            "weight_grid": grid,
            "neighbor_grid": neighbor_grid,
            "model_spec_count": len(standard_models),
        }
    elif search_profile == "formal_v2_extended":
        low_grid = [0.0, 0.01, 0.0625, 0.25, 1.0]
        signal_grid = [1.0, 4.0, 16.0, 64.0, 256.0]
        extended_neighbors = [3, 5, 7, 9, 15]
        for low, cache, short_p99, neighbors, weighting, normalize in itertools.product(
                low_grid, signal_grid, signal_grid, extended_neighbors,
                ("uniform", "inverse_distance"), (False, True)):
            weights = {
                "shape": 1.0, "arrival": low, "queue": low, "kv": low,
                "cache": cache, "short_p99": short_p99,
            }
            model = {
                "selector": "telemetry_aware_knn_cost",
                "neighbor_count": neighbors,
                "neighbor_weighting": weighting,
                "normalize_training_cost_by_oracle": normalize,
            }
            trial_inputs.append((weights, model))
        search_space = {
            "weak_signal_shared_weight_grid": low_grid,
            "cache_weight_grid": signal_grid,
            "short_p99_weight_grid": signal_grid,
            "neighbor_grid": extended_neighbors,
            "neighbor_weighting": ["uniform", "inverse_distance"],
            "normalize_training_cost_by_oracle": [False, True],
            "说明": "所有组合仅读取训练 aggregate；独立测试 outcome 未读取。",
        }
    elif search_profile == "formal_v3_regime_rules":
        if template.get("selector") != "telemetry_aware_regime_rules":
            raise ValueError("v3 regime 搜索必须使用 regime selector 模板")
        trial_inputs.append((
            {group: 1.0 for group in GROUPS},
            {"selector": "telemetry_aware_regime_rules"},
        ))
        search_space = {
            "model": "pre_registered_architecture_aware_regime_rules",
            "trial_count": 1,
            "rule_ids": [rule["id"] for rule in template["regime_rules"]],
            "说明": (
                "规则由 v2 失败的训练侧误差分析产生并作为 v3 新协议冻结；"
                "所有条件只使用决策时可观测字段，独立测试 outcome 未读取。"
            ),
        }
    else:
        raise ValueError(f"未知搜索配置：{search_profile}")

    trials = []
    invalid_reasons: Counter[str] = Counter()
    for weights, model in trial_inputs:
        config = weighted_config(template, weights)
        config.update(model)
        try:
            cv = evaluate_cv(rows, config)
        except ValueError as error:
            invalid_reasons[str(error)] += 1
            continue
        metrics = cv["metrics"]
        trials.append({"weights": weights, "model": model, "metrics": metrics})
    if not trials:
        reasons = "; ".join(
            f"{count}× {reason}" for reason, count in invalid_reasons.most_common(5))
        raise ValueError(f"没有可评估的权重组合；原因：{reasons}")
    best = min(trials, key=lambda row: (
        row["metrics"]["p95_regret_pct"],
        row["metrics"]["median_regret_pct"],
        -row["metrics"]["top1_accuracy"],
        row["model"].get("neighbor_count", 0),
        row["model"].get("neighbor_weighting", ""),
        row["model"].get("normalize_training_cost_by_oracle", False),
        tuple(row["weights"][key] for key in tunable),
    ))
    fitted = weighted_config(template, best["weights"])
    fitted.update(best["model"])
    best_cv = evaluate_cv(rows, fitted)
    worst_rows = sorted(
        best_cv["rows"], key=lambda row: row["regret_pct"], reverse=True)[:20]
    fold_metrics = {}
    for held_out in best_cv["folds"]:
        fold_rows = [row for row in best_cv["rows"]
                     if row["held_out_family"] == held_out]
        regrets = [row["regret_pct"] for row in fold_rows]
        fold_metrics[held_out] = {
            "rows": len(fold_rows),
            "median_regret_pct": statistics.median(regrets),
            "p95_regret_pct": percentile(regrets, 0.95),
            "maximum_regret_pct": max(regrets),
        }
    thresholds = fitted["gates"]
    training_gates = {
        "median_regret": best["metrics"]["median_regret_pct"]
                         <= thresholds["median_regret_pct_max"],
        "p95_regret": best["metrics"]["p95_regret_pct"]
                      <= thresholds["p95_regret_pct_max"],
    }
    fit_method = (
        "leave_one_W_family_out_training_derived_regime_rules_frozen_before_independent_test"
        if search_profile == "formal_v3_regime_rules"
        else "leave_one_W_family_out_cost_sensitive_knn_grid_search"
    )
    fitted.update({
        "status": "frozen" if all(training_gates.values()) else "training_gate_failed",
        "fit_method": fit_method,
        "fit_weights": best["weights"],
        "training_metrics": best["metrics"],
        "training_gates": training_gates,
        "independent_test_read": False,
    })
    report = {
        "schema_version": "qtopomoe.phase8_selector_fit_report.v1",
        "status": fitted["status"],
        "search_profile": search_profile,
        "search_space": search_space,
        "grid": grid,
        "neighbor_grid": neighbor_grid,
        "trial_count": len(trials),
        "invalid_trial_count": sum(invalid_reasons.values()),
        "invalid_trial_reasons": dict(invalid_reasons.most_common(20)),
        "best": best,
        "best_fold_metrics": fold_metrics,
        "best_worst_rows": worst_rows,
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
    parser.add_argument("--neighbor-grid", default="1,3,5,7")
    parser.add_argument(
        "--search-profile", choices=(
            "standard", "formal_v2_extended", "formal_v3_regime_rules"),
        default="standard")
    args = parser.parse_args()
    aggregate = json.loads(args.training_aggregate.read_text(encoding="utf-8"))
    template = json.loads(args.template.read_text(encoding="utf-8"))
    grid = [float(value) for value in args.weight_grid.split(",")]
    if any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("权重网格必须是有限正数")
    neighbor_grid = [int(value) for value in args.neighbor_grid.split(",")]
    if any(value <= 0 for value in neighbor_grid):
        raise ValueError("邻居网格必须为正整数")
    fitted, report = fit(
        aggregate, template, grid, neighbor_grid, args.search_profile)
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
