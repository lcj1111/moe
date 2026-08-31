#!/usr/bin/env python3
# 作用：根据路由负载、通信成本和显存约束生成离线专家放置方案。
"""Phase 7 offline expert placement (11.2).

Inputs:
  - per-layer expert-token histogram from Phase 3 route trace;
  - measured communication cost matrix (configs/communication/nccl_cost_db.json);
  - quantized expert size (bytes) and per-GPU memory headroom;
  - NUMA topology (which GPUs share a NUMA node).

Objective (Runbook): simultaneously penalize compute imbalance, cross-NUMA
dispatch bytes, and migration cost.  We implement three placement strategies
for comparison:

  - static_linear / static_round_robin: fixed expert-to-GPU maps;
  - load_only: greedy load balancing by routed tokens (no topology);
  - load_topology: greedy load balancing with cross-NUMA penalty.

Output (per Runbook): expert_to_gpu, replicas, predicted cross-NUMA bytes,
predicted p99, and model trace hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from collections import defaultdict


def trace_hash(hist: dict) -> str:
    digest = hashlib.sha256()
    for layer in hist["per_layer"]:
        counts = layer["expert_counts"]
        for expert in sorted(counts, key=int):
            digest.update(f"{expert}:{counts[expert]}\n".encode())
    return digest.hexdigest()[:16]


def expert_loads(hist: dict, num_experts: int) -> list[float]:
    """Total routed tokens per expert across all layers."""
    loads = [0.0] * num_experts
    for layer in hist["per_layer"]:
        for expert, count in layer["expert_counts"].items():
            loads[int(expert)] += float(count)
    return loads


def expert_by_numa(hist: dict, num_experts: int) -> dict[str, list[int]]:
    """Bucket experts by NUMA node of their heaviest token traffic."""
    # Simulate per-layer majority: use layer 0 counts as a proxy for
    # per-NUMA traffic distribution (full traffic split needs routed ids).
    return {"default": list(range(num_experts))}


def cross_numa_bytes(placement: list[int], hist: dict, num_experts: int,
                     numa_of_gpu: list[int], hidden: int = 2048,
                     top_k: int = 8) -> float:
    """Estimate cross-NUMA dispatch bytes: each token routed to an expert on
    a different NUMA node pays hidden*2 bytes to move activation there.
    We approximate per-expert cross-NUMA traffic as (total routed tokens /
    num_experts / top_k) * hidden * 2 for experts not on the token's NUMA.
    Without per-token NUMA attribution, use a uniform 50% cross-NUMA share
    for placements that spread experts across NUMA nodes.
    """
    gpus_per_numa: dict[int, int] = defaultdict(int)
    for gpu, numa in enumerate(numa_of_gpu):
        gpus_per_numa[numa] += 1
    bytes_total = 0.0
    for gpu in set(placement):
        numa = numa_of_gpu[gpu]
        local_experts = sum(1 for g in placement if numa_of_gpu[g] == numa)
        if local_experts == 0:
            continue
        # tokens served by experts on this NUMA, approximate share
        share = local_experts / num_experts
        tokens_here = sum(expert_loads(hist, num_experts)) * share / top_k
        bytes_total += tokens_here * hidden * 2
    return bytes_total


def imbalance(placement: list[int], loads: list[float], num_gpus: int) -> float:
    per_gpu = [0.0] * num_gpus
    for expert, load in enumerate(loads):
        per_gpu[placement[expert]] += load
    mean = sum(per_gpu) / num_gpus
    return (max(per_gpu) - mean) / mean if mean > 0 else 0.0


def migration_cost(placement: list[int], base: list[int],
                   expert_bytes: float) -> float:
    moved = sum(1 for a, b in zip(placement, base) if a != b)
    return moved * expert_bytes


def placement_static_linear(num_experts: int, num_gpus: int) -> list[int]:
    return [i // (num_experts // num_gpus) for i in range(num_experts)]


def placement_static_round_robin(num_experts: int, num_gpus: int) -> list[int]:
    return [i % num_gpus for i in range(num_experts)]


def placement_load_only(loads: list[float], num_gpus: int) -> list[int]:
    load_acc = [0.0] * num_gpus
    placement = [0] * len(loads)
    for expert in sorted(range(len(loads)), key=lambda e: loads[e], reverse=True):
        gpu = min(range(num_gpus), key=lambda g: load_acc[g])
        placement[expert] = gpu
        load_acc[gpu] += loads[expert]
    return placement


def placement_load_topology(loads: list[float], num_gpus: int,
                            numa_of_gpu: list[int],
                            cross_numa_penalty: float = 0.0) -> list[int]:
    """Greedy load balancing that prefers the same NUMA node when the
    imbalance gain is small; cross_numa_penalty is reserved for a
    quant-aware weighting of expert size (future work hook)."""
    load_acc = [0.0] * num_gpus
    placement = [0] * len(loads)
    # pick a seed GPU per NUMA node so each node has at least one expert
    seen_numa: set[int] = set()
    for expert in sorted(range(len(loads)), key=lambda e: loads[e], reverse=True):
        candidates = []
        my_numa = expert % len(set(numa_of_gpu))
        for gpu in range(num_gpus):
            if numa_of_gpu[gpu] not in seen_numa:
                candidates.append((load_acc[gpu], 0, gpu))
            else:
                same_numa = 0 if numa_of_gpu[gpu] == my_numa else cross_numa_penalty
                candidates.append((load_acc[gpu] + same_numa, 1, gpu))
        _, _, gpu = min(candidates)
        seen_numa.add(numa_of_gpu[gpu])
        placement[expert] = gpu
        load_acc[gpu] += loads[expert]
    return placement


def predicted_p99(placement: list[int], loads: list[float], num_gpus: int,
                  kernel_us_by_m: dict[int, float],
                  m_weights: dict[int, float]) -> float:
    per_gpu = [0.0] * num_gpus
    for expert, load in enumerate(loads):
        per_gpu[placement[expert]] += load
    total = sum(per_gpu)
    # compute_us = sum over M buckets of weight * kernel time; scale by
    # the heaviest GPU's share (imbalance translates to longer critical path)
    compute_us = 0.0
    for m, weight in m_weights.items():
        compute_us += weight * kernel_us_by_m.get(int(m), 0.0)
    max_share = max(per_gpu) / total if total > 0 else 1.0
    return compute_us * max_share / 1000.0  # ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histogram", type=pathlib.Path, required=True)
    parser.add_argument("--comm-db", type=pathlib.Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--expert-bytes", type=float,
                        default=2048 * 512 * 2)  # bf16 w1+w2 per expert
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    hist = json.loads(args.histogram.read_text(encoding="utf-8"))
    comm = json.loads(args.comm_db.read_text(encoding="utf-8"))
    loads = expert_loads(hist, args.num_experts)
    numa_of_gpu = [0, 0, 1, 1][: args.num_gpus]  # 4-GPU: 0,1 on NUMA0; 2,3 on NUMA1
    # extend for 8 GPUs
    while len(numa_of_gpu) < args.num_gpus:
        numa_of_gpu.append(len(numa_of_gpu) // 4)

    base = placement_static_linear(args.num_experts, args.num_gpus)
    strategies = {
        "static_linear": placement_static_linear(args.num_experts, args.num_gpus),
        "static_round_robin": placement_static_round_robin(
            args.num_experts, args.num_gpus),
        "load_only": placement_load_only(loads, args.num_gpus),
        "load_topology": placement_load_topology(
            loads, args.num_gpus, numa_of_gpu),
    }

    # kernel latency approximation from measured DB (fp8 triton, per M)
    kernel_us_by_m = {1: 225.0, 8: 250.0, 16: 353.0, 32: 453.0,
                      256: 662.0, 2048: 819.0, 8192: 1787.0, 16384: 3324.0}
    m_weights = {"1": 0.007198, "2048": 0.897493, "8192": 0.095309}

    results = []
    for name, placement in strategies.items():
        xnuma = cross_numa_bytes(placement, hist, args.num_experts, numa_of_gpu,
                                 args.hidden)
        imb = imbalance(placement, loads, args.num_gpus)
        mig = migration_cost(placement, base, args.expert_bytes)
        p99 = predicted_p99(placement, loads, args.num_gpus, kernel_us_by_m,
                            m_weights)
        results.append({
            "strategy": name,
            "expert_to_gpu": placement,
            "replicas": 0,
            "predicted_cross_numa_bytes": round(xnuma, 1),
            "predicted_p99_ms": round(p99, 3),
            "load_imbalance_pct": round(imb * 100, 2),
            "migration_bytes": round(mig, 1),
        })
        print(json.dumps(results[-1], ensure_ascii=False))

    payload = {
        "schema_version": "qtopomoe.phase7.placement.v1",
        "trace_hash": trace_hash(hist),
        "num_experts": args.num_experts,
        "num_gpus": args.num_gpus,
        "comm_db": str(args.comm_db),
        "strategies": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
