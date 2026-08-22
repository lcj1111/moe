#!/usr/bin/env python3
"""Normalize formal NCCL statistics into mapping-aware communication costs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LABEL_TO_MAPPING = {
    "pix01": "tp2_pix_0_1", "node02": "tp2_node_0_2", "sys04_bind0": "tp4_sys_0_4",
    "numa0": "tp4_numa0", "numa1": "tp4_numa1", "all8": "tp8_sys",
}


def build(rows: list[dict], min_size_bytes: int = 1 << 20) -> dict:
    points = []
    for row in rows:
        if row.get("metric") != "time_us" or row.get("wrong_total", 1) != 0:
            continue
        label = row.get("label")
        mapping = LABEL_TO_MAPPING.get(label, label)
        size = int(row["size_bytes"])
        time_us = float(row["mean"])
        points.append({"collective": row["collective"], "label": label, "mapping": mapping,
                       "size_bytes": size, "mean_time_us": time_us,
                       "ci95_half_width_us": float(row.get("ci95_half_width", 0)),
                       "n": int(row.get("n", 0)), "wrong_total": int(row.get("wrong_total", 0)),
                       "effective_us_per_gb": time_us / size * 1e9})
    grouped: dict[str, list[dict]] = {}
    for p in points:
        if p["size_bytes"] >= min_size_bytes:
            grouped.setdefault(p["mapping"], []).append(p)
    mapping_summary = []
    for mapping, items in sorted(grouped.items()):
        rates = [p["effective_us_per_gb"] for p in items]
        mapping_summary.append({"mapping": mapping, "large_message_points": len(items),
                                "median_effective_us_per_gb": statistics.median(rates),
                                "min_size_bytes": min(p["size_bytes"] for p in items),
                                "max_size_bytes": max(p["size_bytes"] for p in items)})
    return {"schema_version": "qtopomoe.communication_cost.v1",
            "source": "artifacts/raw/20260804T040000Z_nccl_formal/nccl/statistics.json",
            "metric_definition": "mean NCCL time_us; effective_us_per_gb is a descriptive rate, not a linear model",
            "points": points, "mapping_summary": mapping_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default=None,
                        help="source path recorded in the DB (default: input)")
    args = parser.parse_args()
    result = build(json.loads(args.input.read_text(encoding="utf-8")))
    result["source"] = args.source or str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"normalized {len(result['points'])} time points across {len(result['mapping_summary'])} mappings")


if __name__ == "__main__":
    main()
