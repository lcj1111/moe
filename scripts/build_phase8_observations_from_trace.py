#!/usr/bin/env python3
"""Build Phase 8 workload observations from the real Phase 3 route trace.

The BF16 full trace histogram records, per layer, how many tokens each of the
256 experts received. That per-expert token count is exactly the MoE kernel
M value, so we aggregate it into the plan M buckets and derive:

  - ``real_M_hist``: weight of each M bucket (per-expert token count);
  - ``route_hist``: per-expert share of routed tokens (load distribution);
  - ``communication_bytes``: estimated per-request bytes for a TP4 MoE
    forward (hidden_size*2 bytes per layer per token, 40 layers);
  - ``migration_bytes``: 0 (EPLB disabled in these observations).

Usage (server):
    python3 scripts/build_phase8_observations_from_trace.py \
        --histogram /data/models/test/qtopomoe_traces/v1/bf16_full/expert_token_histogram.json \
        --output /tmp/phase8_obs_from_trace.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


BUCKETS = [1, 8, 16, 32, 256, 2048, 8192, 16384]


def to_bucket(m: int) -> int:
    """Map a raw M to the nearest plan bucket (ceil to next bucket)."""
    for bucket in BUCKETS:
        if m <= bucket:
            return bucket
    return BUCKETS[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histogram", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--communication-bytes-per-token", type=float,
                        default=2.0 * 2048 * 2 * 40)  # hidden*2 bytes * layers
    parser.add_argument("--source-tag", default="phase3_bf16_full_trace")
    args = parser.parse_args()

    hist = json.loads(args.histogram.read_text(encoding="utf-8"))
    bucket_weight: dict[int, float] = defaultdict(float)
    route_total = defaultdict(float)
    total_tokens = 0
    for layer in hist["per_layer"]:
        counts = layer["expert_counts"]
        for expert, count in counts.items():
            m = int(count)
            bucket_weight[to_bucket(m)] += float(m)
            route_total[expert] += float(m)
            total_tokens += float(m)

    real_M_hist = {
        str(bucket): round(weight / total_tokens, 6)
        for bucket, weight in sorted(bucket_weight.items())
    }
    route_hist = {
        expert: round(count / total_tokens, 6)
        for expert, count in sorted(route_total.items())
    }
    # total_tokens sums all 40 layers; per-request tokens use one layer's sum
    # (each layer routes the same token set) over the 116 trace prompts.
    layer_tokens = total_tokens / args.layers
    per_request_tokens = layer_tokens / 116.0
    communication_bytes = per_request_tokens * args.communication_bytes_per_token

    observations = [
        {
            "real_M_hist": real_M_hist,
            "communication_bytes": round(communication_bytes, 1),
            "route_hist": route_hist,
            "placement": {"source": args.source_tag, "trace": str(args.histogram)},
            "migration_bytes": 0,
        }
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(observations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {"real_M_hist": real_M_hist,
         "communication_bytes": communication_bytes,
         "route_hist_entries": len(route_hist)},
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
