#!/usr/bin/env python3
# 作用：从真实 route trace 构造 Phase 8 workload 观测。
"""Build Phase 8 workload observations from the real Phase 3 route trace.

Phase 4/8 ``m_bucket`` semantics: a bucket is a *batch token count* (the MoE
kernel's flat ``num_tokens``), where ``prefill_m = input_tokens * concurrency``
and ``decode_m = concurrency``.  This script derives that distribution from
the trace capture manifest:

  - each prompt contributes one prefill event with M = its prompt token count
    (rounded up to the plan bucket);
  - each generated token contributes one decode event with M = 1
    (concurrency 1 in the capture);

``route_hist`` is the per-expert share of routed tokens (load distribution)
from the expert token histogram; ``communication_bytes`` is an estimated
per-request MoE forward byte count.

Usage (server):
    python3 scripts/build_phase8_observations_from_trace.py \
        --manifest /data/models/test/qtopomoe_traces/v1/bf16_full/capture_manifest.json \
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
    """Map a raw M to the plan bucket (ceil to next bucket)."""
    for bucket in BUCKETS:
        if m <= bucket:
            return bucket
    return BUCKETS[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--histogram", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--communication-bytes-per-token", type=float,
                        default=2.0 * 2048 * 2 * 40)  # hidden*2 bytes * layers
    parser.add_argument("--source-tag", default="phase3_bf16_full_trace")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    hist = json.loads(args.histogram.read_text(encoding="utf-8"))
    gen_tokens = int(manifest.get("gen_tokens", 0))

    # Prefill events: one per prompt, M = prompt token count.
    prefill_weight: dict[int, float] = defaultdict(float)
    for prompt in manifest["prompts"]:
        prefill_weight[to_bucket(int(prompt["prompt_tokens"]))] += 1.0
    # Decode events: one per generated token, M = 1 (concurrency 1).
    decode_weight: dict[int, float] = defaultdict(float)
    decode_weight[to_bucket(1)] = float(len(manifest["prompts"]) * max(gen_tokens, 1))

    # Weight by token count so prefill dominates (token-weighted mixture).
    all_weight: dict[int, float] = defaultdict(float)
    for bucket, count in prefill_weight.items():
        all_weight[bucket] += count * bucket
    for bucket, count in decode_weight.items():
        all_weight[bucket] += count * bucket
    total = sum(all_weight.values())
    real_M_hist = {
        str(bucket): round(weight / total, 6)
        for bucket, weight in sorted(all_weight.items())
    }

    # Route histogram: per-expert share of routed tokens across all layers.
    route_total = defaultdict(float)
    total_tokens = 0
    for layer in hist["per_layer"]:
        for expert, count in layer["expert_counts"].items():
            route_total[expert] += float(count)
            total_tokens += float(count)
    route_hist = {
        expert: round(count / total_tokens, 6)
        for expert, count in sorted(route_total.items())
    }

    per_request_prompt_tokens = sum(
        int(p["prompt_tokens"]) for p in manifest["prompts"]
    ) / len(manifest["prompts"])
    per_request_tokens = per_request_prompt_tokens + gen_tokens
    communication_bytes = per_request_tokens * args.communication_bytes_per_token

    observations = [
        {
            "real_M_hist": real_M_hist,
            "communication_bytes": round(communication_bytes, 1),
            "route_hist": route_hist,
            "placement": {"source": args.source_tag, "trace": str(args.manifest)},
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
         "route_hist_entries": len(route_hist),
         "prefill_events": len(manifest["prompts"]),
         "decode_events": len(manifest["prompts"]) * gen_tokens},
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
