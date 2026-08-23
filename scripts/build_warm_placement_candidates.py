#!/usr/bin/env python3
"""从真实暖态 EPLB 负载窗口生成限制迁移量的 placement 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean > 0 else 0.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def read_warm_counts(path: Path, max_records: int) -> tuple[list[list[float]], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("暖态负载窗口为空")
    selected = records[-max_records:] if max_records > 0 else records
    first = selected[0]["models"][0]
    counts = [[0.0 for _ in row] for row in first]
    model_windows = 0
    for record in selected:
        for model in record["models"]:
            if len(model) != len(counts) or any(
                len(row) != len(counts[index]) for index, row in enumerate(model)
            ):
                raise ValueError("暖态窗口形状不一致")
            for layer, row in enumerate(model):
                for expert, value in enumerate(row):
                    counts[layer][expert] += float(value)
            model_windows += 1
    return counts, {
        "records_total": len(records),
        "records_used": len(selected),
        "model_windows_used": model_windows,
        "first_call": selected[0]["call"],
        "last_call": selected[-1]["call"],
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def identity_map(layers: int, experts: int) -> list[list[int]]:
    return [list(range(experts)) for _ in range(layers)]


def map_metrics(mapping: list[list[int]], counts: list[list[float]], ranks: int) -> dict[str, Any]:
    experts = len(counts[0])
    slots = experts // ranks
    layer_cvs: list[float] = []
    layer_max_over_mean: list[float] = []
    total_by_rank = [0.0] * ranks
    rank_loads_by_layer: list[list[float]] = []
    for layer, row in enumerate(mapping):
        loads = [
            sum(counts[layer][expert] for expert in row[rank * slots:(rank + 1) * slots])
            for rank in range(ranks)
        ]
        rank_loads_by_layer.append(loads)
        layer_cvs.append(coefficient_of_variation(loads))
        mean = statistics.fmean(loads)
        layer_max_over_mean.append(max(loads) / mean if mean else 0.0)
        total_by_rank = [left + right for left, right in zip(total_by_rank, loads)]
    return {
        "layer_rank_cv_median": statistics.median(layer_cvs),
        "layer_rank_cv_p95": percentile(layer_cvs, 0.95),
        "layer_rank_cv_max": max(layer_cvs),
        "layer_rank_max_over_mean_p95": percentile(layer_max_over_mean, 0.95),
        "global_rank_cv": coefficient_of_variation(total_by_rank),
        "global_rank_loads": total_by_rank,
        "layer_rank_cvs": layer_cvs,
        "rank_loads_by_layer": rank_loads_by_layer,
    }


def limited_swap_map(
    counts: list[list[float]], ranks: int, moved_slots_per_layer: int
) -> list[list[int]]:
    experts = len(counts[0])
    slots = experts // ranks
    if moved_slots_per_layer < 0 or moved_slots_per_layer % 2:
        raise ValueError("每层迁移槽位数必须是非负偶数")
    result: list[list[int]] = []
    for layer_counts in counts:
        groups = [list(range(rank * slots, (rank + 1) * slots)) for rank in range(ranks)]
        loads = [sum(layer_counts[expert] for expert in group) for group in groups]
        used: set[int] = set()
        for _ in range(moved_slots_per_layer // 2):
            best: tuple[float, int, int, int, int] | None = None
            for left in range(ranks):
                for right in range(left + 1, ranks):
                    before = loads[left] ** 2 + loads[right] ** 2
                    for left_expert in groups[left]:
                        if left_expert in used:
                            continue
                        for right_expert in groups[right]:
                            if right_expert in used:
                                continue
                            delta = layer_counts[right_expert] - layer_counts[left_expert]
                            after = (loads[left] + delta) ** 2 + (loads[right] - delta) ** 2
                            improvement = before - after
                            candidate = (
                                improvement, -left, -right, -left_expert, -right_expert
                            )
                            if best is None or candidate > best:
                                best = candidate
            if best is None or best[0] <= 0:
                break
            _, neg_left, neg_right, neg_left_expert, neg_right_expert = best
            left, right = -neg_left, -neg_right
            left_expert, right_expert = -neg_left_expert, -neg_right_expert
            left_index = groups[left].index(left_expert)
            right_index = groups[right].index(right_expert)
            groups[left][left_index], groups[right][right_index] = right_expert, left_expert
            delta = layer_counts[right_expert] - layer_counts[left_expert]
            loads[left] += delta
            loads[right] -= delta
            used.update((left_expert, right_expert))
        # 保留未交换专家的原物理槽位，只改动被选中的两个槽位；这样
        # moved_physical_slots 与真实迁移范围一致，不会因 rank 内排序被放大。
        result.append([expert for group in groups for expert in group])
    return result


def full_lpt_map(counts: list[list[float]], ranks: int) -> list[list[int]]:
    experts = len(counts[0])
    slots = experts // ranks
    result: list[list[int]] = []
    for layer_counts in counts:
        groups = [[] for _ in range(ranks)]
        loads = [0.0] * ranks
        for expert in sorted(range(experts), key=lambda item: (-layer_counts[item], item)):
            rank = min(
                (item for item in range(ranks) if len(groups[item]) < slots),
                key=lambda item: (loads[item], item),
            )
            groups[rank].append(expert)
            loads[rank] += layer_counts[expert]
        result.append([expert for group in groups for expert in sorted(group)])
    return result


def moved_slots(mapping: list[list[int]]) -> int:
    return sum(expert != slot for row in mapping for slot, expert in enumerate(row))


def validate_map(mapping: list[list[int]], layers: int, experts: int) -> None:
    if len(mapping) != layers or any(len(row) != experts for row in mapping):
        raise ValueError("候选 map 形状错误")
    expected = list(range(experts))
    for layer, row in enumerate(mapping):
        if sorted(row) != expected:
            raise ValueError(f"第 {layer} 层不是完整逻辑专家排列")


def candidate_document(
    candidate_id: str,
    mapping: list[list[int]],
    metrics: dict[str, Any],
    identity_metrics: dict[str, Any],
    capture: dict[str, Any],
    ranks: int,
) -> dict[str, Any]:
    compact_metrics = {key: value for key, value in metrics.items() if key not in {
        "layer_rank_cvs", "rank_loads_by_layer"
    }}
    return {
        "schema_version": "qtopomoe.runtime_placement.v2",
        "candidate_id": candidate_id,
        "status": "generated_offline_not_yet_online_accepted",
        "生成说明": "基于最终代表负载的真实暖态逐层逻辑专家计数生成；不修改旧 Gate 或旧 placement。",
        "num_layers": len(mapping),
        "num_logical_experts": len(mapping[0]),
        "num_ranks": ranks,
        "slots_per_rank": len(mapping[0]) // ranks,
        "capture": capture,
        "moved_physical_slots_vs_identity": moved_slots(mapping),
        "predicted_warm_metrics": compact_metrics,
        "predicted_improvement_vs_identity": {
            "layer_rank_cv_median_fraction": (
                compact_metrics["layer_rank_cv_median"]
                / identity_metrics["layer_rank_cv_median"]
            ),
            "layer_rank_cv_p95_fraction": (
                compact_metrics["layer_rank_cv_p95"] / identity_metrics["layer_rank_cv_p95"]
            ),
            "global_rank_cv_fraction": (
                compact_metrics["global_rank_cv"] / identity_metrics["global_rank_cv"]
                if identity_metrics["global_rank_cv"] else 0.0
            ),
        },
        "physical_to_logical_map": mapping,
        "physical_to_logical_map_sha256": canonical_sha256(mapping),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-load-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--move-budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--min-median-improvement", type=float, default=0.20)
    parser.add_argument("--min-p95-improvement", type=float, default=0.15)
    args = parser.parse_args()

    counts, capture = read_warm_counts(args.warm_load_jsonl, args.max_records)
    layers, experts = len(counts), len(counts[0])
    if experts % args.num_ranks:
        raise ValueError("逻辑专家数必须能被 rank 数整除")
    identity = identity_map(layers, experts)
    identity_metrics = map_metrics(identity, counts, args.num_ranks)
    maps = {
        f"warm_swap_{budget:03d}_slots_per_layer_v1": limited_swap_map(
            counts, args.num_ranks, budget
        )
        for budget in args.move_budgets
    }
    maps["warm_full_lpt_v1"] = full_lpt_map(counts, args.num_ranks)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    documents: list[dict[str, Any]] = []
    for candidate_id, mapping in maps.items():
        validate_map(mapping, layers, experts)
        metrics = map_metrics(mapping, counts, args.num_ranks)
        document = candidate_document(
            candidate_id, mapping, metrics, identity_metrics, capture, args.num_ranks
        )
        path = args.output_dir / f"{candidate_id}.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        document["file"] = str(path)
        document["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        documents.append(document)

    eligible = [document for document in documents if (
        document["predicted_improvement_vs_identity"]["layer_rank_cv_median_fraction"]
        <= 1 - args.min_median_improvement
        and document["predicted_improvement_vs_identity"]["layer_rank_cv_p95_fraction"]
        <= 1 - args.min_p95_improvement
    )]
    selected = min(
        eligible,
        key=lambda item: (
            item["moved_physical_slots_vs_identity"],
            item["predicted_warm_metrics"]["layer_rank_cv_p95"],
        ),
    ) if eligible else None
    report = {
        "schema_version": "qtopomoe.warm_placement_candidate_generation.v1",
        "status": "accepted" if selected else "rejected",
        "selection_policy": {
            "min_layer_rank_cv_median_improvement": args.min_median_improvement,
            "min_layer_rank_cv_p95_improvement": args.min_p95_improvement,
            "rule": "在满足离线改善门槛的候选中选择相对 identity 迁移槽位最少者",
        },
        "capture": capture,
        "identity_metrics": {key: value for key, value in identity_metrics.items() if key not in {
            "layer_rank_cvs", "rank_loads_by_layer"
        }},
        "candidates": [{key: value for key, value in document.items() if key not in {
            "physical_to_logical_map"
        }} for document in documents],
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate_file": selected["file"] if selected else None,
        "selected_candidate_map_sha256": (
            selected["physical_to_logical_map_sha256"] if selected else None
        ),
    }
    report_path = args.output_dir / "candidate_generation_gate.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
