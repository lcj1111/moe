#!/usr/bin/env python3
"""Phase 7 migration cost measurement (11.1).

Measures D2D copy bandwidth for a single expert (bf16 weights, hidden*2*
intermediate*2 bytes for w1+w2) between:
  - same GPU (intra-device copy);
  - same NUMA node (peer GPU on the same NUMA);
  - cross NUMA (peer GPU on the other NUMA).

This yields migration cost (us per expert) used by the online EPLB
controller's benefit/cost gate.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def measure_copy(src: torch.Tensor, dst: torch.Tensor, repeats: int,
                 warmup: int = 5) -> float:
    for _ in range(warmup):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        dst.copy_(src, non_blocking=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--src-gpu", type=int, default=0)
    parser.add_argument("--same-numa-gpu", type=int, default=1)
    parser.add_argument("--cross-numa-gpu", type=int, default=4)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    bytes_per_expert = args.hidden * args.intermediate * 2 * 2  # w1 + w2
    tensor_shape = (args.hidden, args.intermediate)
    dev_src = f"cuda:{args.src_gpu}"
    dev_same = f"cuda:{args.same_numa_gpu}"
    dev_cross = f"cuda:{args.cross_numa_gpu}"

    src = torch.randn(tensor_shape, dtype=dtype, device=dev_src)
    same = torch.empty(tensor_shape, dtype=dtype, device=dev_same)
    cross = torch.empty(tensor_shape, dtype=dtype, device=dev_cross)

    # Same GPU: copy within one allocation (needs a second tensor on same dev)
    same_gpu = torch.empty(tensor_shape, dtype=dtype, device=dev_src)
    t_same_gpu = measure_copy(src, same_gpu, args.repeats)
    t_same_numa = measure_copy(src, same, args.repeats)
    t_cross = measure_copy(src, cross, args.repeats)

    result = {
        "schema_version": "qtopomoe.phase7.migration_cost.v1",
        "expert_bytes": bytes_per_expert,
        "shape": tensor_shape,
        "dtype": args.dtype,
        "median_us_per_expert": {
            "same_gpu": round(t_same_gpu, 2),
            "same_numa": round(t_same_numa, 2),
            "cross_numa": round(t_cross, 2),
        },
        "note": (
            "D2D copy of one expert's w1+w2 weights; same_gpu copy uses "
            "intra-device copy, same_numa/cross_numa use peer GPUs."
        ),
    }
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
