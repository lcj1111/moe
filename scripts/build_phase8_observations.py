#!/usr/bin/env python3
"""Bind Runbook workload shapes, route trace, and measured oracle latency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-buckets", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--route-template", type=Path, required=True)
    parser.add_argument("--route-token-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bucket_data = json.loads(args.m_buckets.read_text(encoding="utf-8"))
    screen = json.loads(args.screen_summary.read_text(encoding="utf-8"))
    template_rows = json.loads(args.route_template.read_text(encoding="utf-8"))
    if len(template_rows) != 1:
        raise SystemExit("route template must contain exactly one observation")
    template = template_rows[0]
    if screen.get("status") != "accepted":
        raise SystemExit("screen summary Gate is not accepted")
    if args.route_token_count <= 0:
        raise SystemExit("route token count must be positive")

    m_records = {
        (row["input_tokens"], row["output_tokens"], row["concurrency"]): row
        for row in bucket_data["records"]
        if row["prefix_cache_pct"] == 0 and row["arrival_mode"] == "closed_loop"
    }
    bytes_per_trace_token = float(template["communication_bytes"]) / args.route_token_count
    observations = []
    for screen_row in screen["rows"]:
        key = (screen_row["input_tokens"], screen_row["output_tokens"],
               screen_row["concurrency"])
        if key not in m_records:
            raise SystemExit(f"missing M-bucket record for {screen_row['workload_id']}")
        m_record = m_records[key]
        active_window_tokens = ((screen_row["input_tokens"] + screen_row["output_tokens"])
                                * screen_row["concurrency"])
        observations.append({
            "real_M_hist": m_record["real_M_hist"],
            "communication_bytes": bytes_per_trace_token * active_window_tokens,
            "route_hist": template["route_hist"],
            "placement": {
                "source": "phase8_single_pass_screen_plus_phase3_route_trace",
                "workload_id": screen_row["workload_id"],
                "input_tokens": screen_row["input_tokens"],
                "output_tokens": screen_row["output_tokens"],
                "concurrency": screen_row["concurrency"],
                "prefix_cache_pct": 0,
                "arrival_mode": "closed_loop",
                "prefill_m_bucket": m_record["prefill_m_bucket"],
                "decode_m_bucket": m_record["decode_m_bucket"],
                "communication_estimator": (
                    "phase3_trace_bytes_per_token * active_window_tokens"),
            },
            "migration_bytes": 0,
            "measured_p99_ms_by_candidate": {
                candidate_id: values["e2e_p99_ms"]
                for candidate_id, values in screen_row["measurements"].items()
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observations, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"observation_count": len(observations),
                      "bytes_per_trace_token": bytes_per_trace_token}))


if __name__ == "__main__":
    main()
