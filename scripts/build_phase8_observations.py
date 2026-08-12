#!/usr/bin/env python3
"""Bind Runbook workload shapes, route trace, and measured oracle latency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _measurement_median_and_ci(values: Mapping[str, Any]) -> tuple[float, dict[str, Any] | None]:
    """Read either a single-pass scalar or repeated-bootstrap measurement."""
    if "metrics" not in values:
        return float(values["e2e_p99_ms"]), None
    metric = values["metrics"]["e2e_p99_ms"]
    ci = {
        "low_ms": float(metric["bootstrap_95ci_low"]),
        "high_ms": float(metric["bootstrap_95ci_high"]),
        "n": int(metric["n"]),
        "bootstrap_samples": int(metric["bootstrap_samples"]),
    }
    return float(metric["median"]), ci


def build_observations(bucket_data: Mapping[str, Any], screen: Mapping[str, Any],
                       template: Mapping[str, Any], route_token_count: int) -> list[dict[str, Any]]:
    if screen.get("status") != "accepted":
        raise ValueError("screen summary Gate is not accepted")
    if route_token_count <= 0:
        raise ValueError("route token count must be positive")

    m_records = {
        (row["input_tokens"], row["output_tokens"], row["concurrency"]): row
        for row in bucket_data["records"]
        if row["prefix_cache_pct"] == 0 and row["arrival_mode"] == "closed_loop"
    }
    bytes_per_trace_token = float(template["communication_bytes"]) / route_token_count
    observations = []
    for screen_row in screen["rows"]:
        key = (screen_row["input_tokens"], screen_row["output_tokens"],
               screen_row["concurrency"])
        if key not in m_records:
            raise ValueError(f"missing M-bucket record for {screen_row['workload_id']}")
        m_record = m_records[key]
        active_window_tokens = ((screen_row["input_tokens"] + screen_row["output_tokens"])
                                * screen_row["concurrency"])
        medians: dict[str, float] = {}
        intervals: dict[str, dict[str, Any]] = {}
        for candidate_id, values in screen_row["measurements"].items():
            median, ci = _measurement_median_and_ci(values)
            medians[candidate_id] = median
            if ci is not None:
                intervals[candidate_id] = ci
        placement = {
            "source": "phase8_service_summary_plus_phase3_route_trace",
            "source_schema_version": screen.get("schema_version"),
            "workload_id": screen_row["workload_id"],
            "input_tokens": screen_row["input_tokens"],
            "output_tokens": screen_row["output_tokens"],
            "concurrency": screen_row["concurrency"],
            "base_cell_id": screen_row.get("base_cell_id"),
            "prefix_cache_pct": screen_row.get("prefix_cache_pct", 0),
            "arrival_mode": screen_row.get("arrival_mode", "closed_loop"),
            "prefill_m_bucket": m_record["prefill_m_bucket"],
            "decode_m_bucket": m_record["decode_m_bucket"],
            "communication_estimator": (
                "phase3_trace_bytes_per_token * active_window_tokens"),
        }
        if screen_row.get("request_rate_rps") is not None:
            placement["request_rate_rps"] = float(screen_row["request_rate_rps"])
        if intervals:
            placement["measured_p99_bootstrap_95ci_by_candidate"] = intervals
        observations.append({
            "real_M_hist": m_record["real_M_hist"],
            "communication_bytes": bytes_per_trace_token * active_window_tokens,
            "route_hist": template["route_hist"],
            "placement": placement,
            "migration_bytes": 0,
            "measured_p99_ms_by_candidate": medians,
        })
    return observations


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
    try:
        observations = build_observations(
            bucket_data, screen, template, args.route_token_count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observations, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"observation_count": len(observations),
                      "bytes_per_trace_token": (
                          float(template["communication_bytes"]) / args.route_token_count)}))


if __name__ == "__main__":
    main()
