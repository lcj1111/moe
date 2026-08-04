#!/usr/bin/env python3
"""Generate reproducible Phase 4/8 workloads and MoE M-bucket metadata.

The generator is intentionally framework-neutral.  ``prefill_m`` is the
number of active prefill tokens (input length x concurrency), while
``decode_m`` is the number of concurrent decode rows.  Both are rounded to a
power-of-two bucket so kernel measurements can be indexed consistently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

DEFAULT_M_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_WORKLOADS = {
    "W1": (256, 128, (1, 8, 32)),
    "W2": (2048, 256, (1, 8, 32)),
    "W3": (8192, 256, (1, 8, 16)),
    "W4": (32768, 128, (1,)),
}


def bucket(value: int, buckets: Iterable[int] = DEFAULT_M_BUCKETS) -> int:
    """Return the smallest configured bucket >= value, clamping at max."""
    if value < 1:
        raise ValueError("M must be positive")
    ordered = tuple(sorted(set(int(x) for x in buckets if int(x) > 0)))
    if not ordered:
        raise ValueError("at least one positive bucket is required")
    for candidate in ordered:
        if value <= candidate:
            return candidate
    return ordered[-1]


def build_records(seed: int = 42, buckets: Iterable[int] = DEFAULT_M_BUCKETS) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for workload_id, (input_tokens, output_tokens, concurrencies) in DEFAULT_WORKLOADS.items():
        for concurrency in concurrencies:
            for prefix_cache_pct in (0, 50, 100):
                for arrival_mode in ("closed_loop", "poisson", "burst"):
                    prefill_m = input_tokens * concurrency
                    # A stable seed is recorded so a client can reconstruct burst/Poisson order.
                    stream_seed = rng.randrange(0, 2**32)
                    records.append({
                        "workload_id": workload_id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "concurrency": concurrency,
                        "prefix_cache_pct": prefix_cache_pct,
                        "arrival_mode": arrival_mode,
                        "prefill_m": prefill_m,
                        "decode_m": concurrency,
                        "prefill_m_bucket": bucket(prefill_m, buckets),
                        "decode_m_bucket": bucket(concurrency, buckets),
                        "real_M_hist": {
                            str(bucket(prefill_m, buckets)): 0.75,
                            str(bucket(concurrency, buckets)): 0.25,
                        },
                        "stream_seed": stream_seed,
                    })
    return records


def manifest(seed: int, buckets: Iterable[int]) -> dict:
    rows = build_records(seed, buckets)
    payload = {"schema_version": "qtopomoe.m_bucket.v1", "seed": seed,
               "m_buckets": list(sorted(set(buckets))), "records": rows}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m-buckets", default=",".join(map(str, DEFAULT_M_BUCKETS)))
    args = parser.parse_args()
    buckets = tuple(int(x) for x in args.m_buckets.split(",") if x.strip())
    data = manifest(args.seed, buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(data['records'])} records to {args.output}; sha256={data['sha256']}")


if __name__ == "__main__":
    main()
