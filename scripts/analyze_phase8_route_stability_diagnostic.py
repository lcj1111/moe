#!/usr/bin/env python3
# 作用：聚合逻辑专家分布稳定性诊断并按冻结规则归因。
"""聚合逻辑专家分布稳定性诊断并按预注册规则归因。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return float(ordered[index])


def coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.mean(values) if values else 0.0
    return statistics.pstdev(values) / mean if mean > 0 else 0.0


def add_models(
    target: list[list[list[float]]] | None,
    source: list[list[list[float]]],
) -> list[list[list[float]]]:
    if target is None:
        return [[[float(value) for value in layer] for layer in model] for model in source]
    if len(target) != len(source):
        raise ValueError("诊断记录的模型数量不一致")
    for target_model, source_model in zip(target, source):
        if len(target_model) != len(source_model):
            raise ValueError("诊断记录的 MoE 层数不一致")
        for target_layer, source_layer in zip(target_model, source_model):
            if len(target_layer) != len(source_layer):
                raise ValueError("诊断记录的逻辑专家数量不一致")
            for index, value in enumerate(source_layer):
                target_layer[index] += float(value)
    return target


def distribution_distance(
    left: list[list[list[float]]],
    right: list[list[list[float]]],
) -> dict[str, Any]:
    tv = []
    cv_ratios = []
    for left_model, right_model in zip(left, right):
        for left_layer, right_layer in zip(left_model, right_model):
            left_total = sum(left_layer)
            right_total = sum(right_layer)
            if left_total <= 0 or right_total <= 0:
                continue
            left_probability = [value / left_total for value in left_layer]
            right_probability = [value / right_total for value in right_layer]
            tv.append(
                0.5
                * sum(
                    abs(left_value - right_value)
                    for left_value, right_value in zip(
                        left_probability, right_probability
                    )
                )
            )
            left_cv = coefficient_of_variation(left_layer)
            right_cv = coefficient_of_variation(right_layer)
            if left_cv > 0:
                cv_ratios.append(right_cv / left_cv)
    return {
        "layers": len(tv),
        "tv_median": statistics.median(tv) if tv else 0.0,
        "tv_p95": percentile(tv, 0.95),
        "tv_max": max(tv) if tv else 0.0,
        "cv_ratio_median": statistics.median(cv_ratios) if cv_ratios else 1.0,
        "tv_by_layer": tv,
    }


def request_audit(root: Path, phases: list[str]) -> dict[str, Any]:
    phase_rows = {}
    for phase in phases:
        rows = [
            json.loads(line)
            for line in (root / phase / "requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        phase_rows[phase] = {int(row["request_id"]): row for row in rows}
    common = sorted(set.intersection(*(set(rows) for rows in phase_rows.values())))
    prompts_match = all(
        len({phase_rows[phase][item]["prompt_sha256"] for phase in phases}) == 1
        for item in common
    )
    pair_matches = {}
    for left_index, left in enumerate(phases):
        for right in phases[left_index + 1 :]:
            pair_matches[f"{left}__{right}"] = sum(
                phase_rows[left][item].get("response_sha256")
                == phase_rows[right][item].get("response_sha256")
                for item in common
            )
    return {
        "records_per_phase": {phase: len(rows) for phase, rows in phase_rows.items()},
        "common_request_ids": len(common),
        "prompts_match": prompts_match,
        "cached_tokens_by_phase": {
            phase: sorted(
                {
                    int(row.get("cached_tokens", 0))
                    for row in phase_rows[phase].values()
                }
            )
            for phase in phases
        },
        "response_hash_matches": pair_matches,
    }


def analyze(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (root / "route_diagnostic_windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ranges = load(root / "phase_ranges.json")
    phase_stats: dict[str, Any] = {}
    aggregates: dict[str, list[list[list[float]]]] = {}
    transition_tv = []
    steady_tv = []
    transition_mixed_slots = 0
    steady_mixed_slots = 0
    for item in ranges["phases"]:
        name = str(item["phase"])
        selected = records[int(item["start_line"]) : int(item["end_line"])]
        legacy_tv = [
            float(value)
            for row in selected
            for model in row["legacy_vs_record_time_tv_by_model"]
            for value in model
        ]
        mixed = sum(
            int(value)
            for row in selected
            for value in row["record_time_mismatched_slots"]
        )
        aggregate = None
        for row in selected:
            aggregate = add_models(aggregate, row["record_time_logical_models"])
        phase_stats[name] = {
            "arm": item["arm"],
            "role": item["role"],
            "diagnostic_records": len(selected),
            "record_time_mismatched_slots": mixed,
            "legacy_vs_record_time_tv_median": (
                statistics.median(legacy_tv) if legacy_tv else 0.0
            ),
            "legacy_vs_record_time_tv_p95": percentile(legacy_tv, 0.95),
            "legacy_vs_record_time_tv_max": max(legacy_tv) if legacy_tv else 0.0,
            "current_map_sha256": sorted(
                {
                    value
                    for row in selected
                    for value in row["current_map_sha256"]
                }
            ),
        }
        if aggregate is not None:
            aggregates[name] = aggregate
        if item["role"] == "transition":
            transition_tv.extend(legacy_tv)
            transition_mixed_slots += mixed
        elif item["role"] in {"settle", "measured"}:
            steady_tv.extend(legacy_tv)
            steady_mixed_slots += mixed

    thresholds = plan["分类阈值"]
    arm_results = {}
    for arm_name, arm in plan["诊断负载"].items():
        phases = list(arm["测量阶段"])
        identity_a1, candidate_b1, identity_a2, candidate_b2 = phases
        identity_repeat = distribution_distance(
            aggregates[identity_a1], aggregates[identity_a2]
        )
        candidate_repeat = distribution_distance(
            aggregates[candidate_b1], aggregates[candidate_b2]
        )
        cross_rows = [
            distribution_distance(aggregates[identity], aggregates[candidate])
            for identity in (identity_a1, identity_a2)
            for candidate in (candidate_b1, candidate_b2)
        ]
        cross_tv = [value for row in cross_rows for value in row["tv_by_layer"]]
        cross_p95 = percentile(cross_tv, 0.95)
        repeat_baseline_p95 = max(
            float(identity_repeat["tv_p95"]), float(candidate_repeat["tv_p95"])
        )
        arm_results[arm_name] = {
            "request_audit": request_audit(root, phases),
            "identity_repeat": identity_repeat,
            "candidate_repeat": candidate_repeat,
            "identity_candidate_cross_tv_median": (
                statistics.median(cross_tv) if cross_tv else 0.0
            ),
            "identity_candidate_cross_tv_p95": cross_p95,
            "repeat_baseline_tv_p95": repeat_baseline_p95,
            "placement_excess_tv_p95": max(0.0, cross_p95 - repeat_baseline_p95),
            "cross_comparisons": [
                {key: value for key, value in row.items() if key != "tv_by_layer"}
                for row in cross_rows
            ],
        }

    steady_legacy_p95 = percentile(steady_tv, 0.95)
    transition_legacy_p95 = percentile(transition_tv, 0.95)
    statistical_scope = steady_legacy_p95 > float(
        thresholds["legacy_vs_record_time_tv_p95_max"]
    )
    counter_reset = (
        transition_mixed_slots > 0
        or transition_legacy_p95
        > float(thresholds["legacy_vs_record_time_tv_p95_max"])
    ) and not statistical_scope
    short_excess = float(arm_results["短输出同输入对照"]["placement_excess_tv_p95"])
    long_excess = float(arm_results["长输出复现对照"]["placement_excess_tv_p95"])
    actual_route_change = short_excess > float(
        thresholds["短输出placement_excess_tv_p95_max"]
    )
    continuation_divergence = (
        not actual_route_change
        and long_excess > float(thresholds["长输出placement_excess_tv_p95_max"])
    )
    confirmation = plan.get("确认归因", {})
    cache_states = [
        tuple(values)
        for arm in arm_results.values()
        for values in arm["request_audit"]["cached_tokens_by_phase"].values()
    ]
    measured_cache_state_matches = len(set(cache_states)) == 1
    sampling_scope = bool(confirmation.get("enabled")) and (
        not statistical_scope
        and not counter_reset
        and not actual_route_change
        and not continuation_divergence
        and measured_cache_state_matches
    )
    if statistical_scope:
        primary_cause = "统计还原口径错误：稳态窗口的旧口径与记录时逻辑计数不一致"
    elif counter_reset:
        primary_cause = "计数器或滑动窗口跨 map 污染：差异只出现在切换过渡窗口"
    elif actual_route_change:
        primary_cause = "实际路由变化：短输出同输入对照仍存在超出重复基线的 placement 差异"
    elif continuation_divergence:
        primary_cause = "长生成轨迹分叉造成的实际路由采样差异，不是统计口径或计数器重置"
    elif sampling_scope:
        primary_cause = str(confirmation["稳定时归因"])
    else:
        primary_cause = "未复现 14.28% 差异；新诊断下统计口径、计数器和实际路由均稳定"

    integrity = {
        "diagnostic_records_present_for_every_phase": all(
            int(value["diagnostic_records"]) > 0 for value in phase_stats.values()
        ),
        "all_measured_prompts_match": all(
            value["request_audit"]["prompts_match"] for value in arm_results.values()
        ),
        "steady_windows_have_no_mixed_map_slots": steady_mixed_slots == 0,
        "finite_metrics": all(
            math.isfinite(float(value["placement_excess_tv_p95"]))
            for value in arm_results.values()
        ),
        "measured_cache_state_matches": measured_cache_state_matches,
    }
    return {
        "schema_version": "qtopomoe.phase8_route_stability_diagnostic.v1",
        "status": "classified" if all(integrity.values()) else "invalid",
        "integrity_gates": integrity,
        "classification": {
            "primary_cause": primary_cause,
            "statistical_scope_or_reconstruction_error": statistical_scope
            or sampling_scope,
            "sampling_or_phase_order_scope": sampling_scope,
            "counter_reset_or_mixed_window": counter_reset,
            "actual_route_change_under_short_matched_input": actual_route_change,
            "long_generation_continuation_divergence": continuation_divergence,
        },
        "legacy_vs_record_time": {
            "transition_tv_p95": transition_legacy_p95,
            "steady_tv_p95": steady_legacy_p95,
            "transition_mismatched_slots": transition_mixed_slots,
            "steady_mismatched_slots": steady_mixed_slots,
        },
        "arms": arm_results,
        "phases": phase_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(load(args.plan), args.output_root)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "classified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
