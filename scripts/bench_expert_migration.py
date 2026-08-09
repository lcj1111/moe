#!/usr/bin/env python3
"""Measure exact-size expert remap/copy primitives on an idle GPU host."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metadata_remap(repeats: int, iterations: int) -> dict[str, object]:
    samples = []
    placement = {"0:0": (0,)}
    for repeat in range(repeats):
        start = time.perf_counter_ns()
        for index in range(iterations):
            placement["0:0"] = (index & 7,)
        elapsed_us = (time.perf_counter_ns() - start) / 1000 / iterations
        samples.append(elapsed_us)
    return {
        "label": "same_gpu_metadata_remap",
        "src_gpu": 0,
        "dst_gpu": 0,
        "migration_bytes": 0,
        "samples_us": samples,
        "mean_us": statistics.fmean(samples),
        "p95_us": percentile(samples, 0.95),
        "verification": placement["0:0"][0] in range(8),
    }


def peer_copy(label: str, src_gpu: int, dst_gpu: int, size_bytes: int,
              repeats: int, iterations: int, warmup: int) -> dict[str, object]:
    with torch.cuda.device(src_gpu):
        src = torch.full((size_bytes,), 7, dtype=torch.uint8, device=f"cuda:{src_gpu}")
    with torch.cuda.device(dst_gpu):
        dst = torch.empty((size_bytes,), dtype=torch.uint8, device=f"cuda:{dst_gpu}")
        stream = torch.cuda.Stream(device=dst_gpu)
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(iterations):
                    dst.copy_(src, non_blocking=True)
                end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / iterations)
        verified = int(dst[0].item()) == 7 and int(dst[-1].item()) == 7
    mean_us = statistics.fmean(samples)
    return {
        "label": label,
        "src_gpu": src_gpu,
        "dst_gpu": dst_gpu,
        "migration_bytes": size_bytes,
        "samples_us": samples,
        "mean_us": mean_us,
        "p95_us": percentile(samples, 0.95),
        "effective_gbps": size_bytes / (mean_us * 1000),
        "peer_access": src_gpu == dst_gpu or torch.cuda.can_device_access_peer(src_gpu, dst_gpu),
        "verification": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
        raise SystemExit("at least five CUDA devices are required")

    rows = [metadata_remap(args.repeats, args.iterations * 100)]
    rows.extend([
        peer_copy("same_gpu_copy", 0, 0, args.expert_bytes, args.repeats, args.iterations, args.warmup),
        peer_copy("same_numa_pix_copy", 0, 1, args.expert_bytes, args.repeats, args.iterations, args.warmup),
        peer_copy("same_numa_node_copy", 0, 2, args.expert_bytes, args.repeats, args.iterations, args.warmup),
        peer_copy("cross_numa_sys_copy", 0, 4, args.expert_bytes, args.repeats, args.iterations, args.warmup),
    ])
    result = {
        "schema_version": "qtopomoe.expert_migration_microbench.v1",
        "expert_bytes": args.expert_bytes,
        "repeats": args.repeats,
        "iterations_per_repeat": args.iterations,
        "rows": rows,
        "all_verified": all(row["verification"] for row in rows),
        "service_impact": {
            "block_time_ms": None,
            "recovery_time_ms": None,
            "affected_p99_ms": None,
            "formal_ready": False,
            "reason": "requires controlled live-service migration experiment",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
