#!/usr/bin/env python3
"""Convert Q-TopoMoE's offline expert placement into vLLM EPLB slot order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--histogram", type=Path)
    parser.add_argument("--repair-with-native-policy", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    assignments = source["plan"]["expert_to_gpu"]
    parsed = [(tuple(map(int, key.split(":"))), value) for key, value in assignments.items()]
    layers = max(layer for (layer, _), _ in parsed) + 1
    logical_experts = max(expert for (_, expert), _ in parsed) + 1
    if logical_experts % args.num_ranks:
        raise ValueError("logical expert count must be divisible by rank count")
    slots_per_rank = logical_experts // args.num_ranks

    by_layer_rank = [[[] for _ in range(args.num_ranks)] for _ in range(layers)]
    for (layer, expert), ranks in parsed:
        if len(ranks) != 1:
            raise ValueError(f"expected exactly one rank for {layer}:{expert}, got {ranks}")
        rank = int(ranks[0])
        if not 0 <= rank < args.num_ranks:
            raise ValueError(f"rank out of range for {layer}:{expert}: {rank}")
        by_layer_rank[layer][rank].append(expert)

    original_counts = [[len(experts) for experts in groups] for groups in by_layer_rank]
    runtime_feasible = all(
        count == slots_per_rank for counts in original_counts for count in counts
    )
    repair_method = "none"
    if runtime_feasible:
        physical_to_logical_map = [
            [expert for experts in groups for expert in sorted(experts)]
            for groups in by_layer_rank
        ]
    else:
        if not args.repair_with_native_policy or not args.histogram:
            worst = max(
                (abs(count - slots_per_rank), layer, rank, count)
                for layer, counts in enumerate(original_counts)
                for rank, count in enumerate(counts)
            )
            _, layer, rank, count = worst
            raise ValueError(
                f"offline plan is not runtime-feasible: layer {layer} rank {rank} "
                f"has {count} experts, expected {slots_per_rank}; supply --histogram "
                "and --repair-with-native-policy"
            )
        import torch
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

        histogram = json.loads(args.histogram.read_text(encoding="utf-8"))
        counts_by_layer = {
            int(item["layer_id"]): item["expert_counts"] for item in histogram["per_layer"]
        }
        weights = torch.tensor(
            [
                [float(counts_by_layer[layer].get(str(expert), 0)) for expert in range(logical_experts)]
                for layer in range(layers)
            ],
            dtype=torch.float32,
        )
        physical_to_logical_map = DefaultEplbPolicy.rebalance_experts(
            weights,
            logical_experts,
            1,
            1,
            args.num_ranks,
        ).tolist()
        repair_method = "vllm_default_eplb_policy_from_measured_histogram"

    initial = [list(range(logical_experts)) for _ in range(layers)]
    moved_slots = sum(
        actual != original
        for row, old_row in zip(physical_to_logical_map, initial)
        for actual, original in zip(row, old_row)
    )
    offline_rank = {
        (layer, expert): int(ranks[0]) for (layer, expert), ranks in parsed
    }
    changed_rank_vs_offline = sum(
        offline_rank[(layer, expert)] != slot // slots_per_rank
        for layer, row in enumerate(physical_to_logical_map)
        for slot, expert in enumerate(row)
    )
    result = {
        "schema_version": 1,
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "num_layers": layers,
        "num_logical_experts": logical_experts,
        "num_ranks": args.num_ranks,
        "slots_per_rank": slots_per_rank,
        "offline_plan_runtime_feasible": runtime_feasible,
        "offline_expert_counts_per_layer_rank": original_counts,
        "repair_method": repair_method,
        "repair_histogram": str(args.histogram) if args.histogram else None,
        "changed_rank_assignments_vs_offline": changed_rank_vs_offline,
        "moved_physical_slots_vs_identity": moved_slots,
        "physical_to_logical_map": physical_to_logical_map,
    }
    result["physical_to_logical_map_sha256"] = canonical_sha256(physical_to_logical_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "physical_to_logical_map"}, indent=2))


if __name__ == "__main__":
    main()
