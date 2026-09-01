#!/usr/bin/env python3
# 作用：统计 Phase 1 已完成矩阵行并输出服务性能摘要。
"""Statistically summarize completed BF16/FP8 Phase 1 matrix rows.

Rows are considered runnable/completed from ``failed`` and ``completed``;
the historical ``acceptance`` field is reported but is not used as a failure
signal because older aggregate files did not carry the acceptance artifact.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "e2e_p50_ms", "e2e_p95_ms")


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected a list in {path}")
        rows.extend(data)
    return rows


def summarize(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get("failed", 0) == 0 and r.get("completed", 0) == r.get("requests", 0)]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in completed:
        key = (row.get("format"), row.get("topology"), row.get("workload"), row.get("concurrency"))
        groups[key].append(row)
    grouped = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        grouped.append({"format": key[0], "topology": key[1], "workload": key[2], "concurrency": key[3],
                        "runs": len(items), **{m: statistics.median([float(x[m]) for x in items if x.get(m) is not None]) for m in METRICS}})
    by_format = {}
    for fmt in sorted({r.get("format") for r in rows}):
        subset = [r for r in grouped if r["format"] == fmt]
        by_format[fmt] = {"rows": len(subset), "completed_requests": sum(r.get("completed", 0) for r in rows if r.get("format") == fmt and r in completed),
                          "best_e2e_p95": min(subset, key=lambda r: r["e2e_p95_ms"]) if subset else None,
                          "best_ttft_p95": min(subset, key=lambda r: r["ttft_p95_ms"]) if subset else None}
    overlap = []
    keys = {(r["topology"], r["workload"], r["concurrency"]) for r in grouped}
    for topology, workload, concurrency in sorted(keys):
        pair = [r for r in grouped if (r["topology"], r["workload"], r["concurrency"]) == (topology, workload, concurrency)]
        if {r["format"] for r in pair} == {"bf16", "fp8"}:
            b, f = next(r for r in pair if r["format"] == "bf16"), next(r for r in pair if r["format"] == "fp8")
            overlap.append({"topology": topology, "workload": workload, "concurrency": concurrency,
                            "e2e_p95_delta_pct_fp8_vs_bf16": (f["e2e_p95_ms"] / b["e2e_p95_ms"] - 1) * 100,
                            "ttft_p95_delta_pct_fp8_vs_bf16": (f["ttft_p95_ms"] / b["ttft_p95_ms"] - 1) * 100})
    return {"row_count": len(rows), "completed_row_count": len(completed),
            "failed_or_incomplete_row_count": len(rows) - len(completed),
            "acceptance_true_count": sum(bool(r.get("acceptance")) for r in rows),
            "grouped": grouped, "by_format": by_format, "overlap": overlap}


def markdown(summary: dict) -> str:
    lines = ["# Q-TopoMoE Phase 1 BF16/FP8 statistical analysis", "", 
             "Completed means `failed == 0` and `completed == requests`; the legacy `acceptance` flag is reported but not used to discard measurements.", "",
             f"- Matrix rows: {summary['row_count']} (completed: {summary['completed_row_count']}, failed/incomplete: {summary['failed_or_incomplete_row_count']})",
             f"- acceptance=true rows: {summary['acceptance_true_count']}", "", "## Grouped medians", "",
             "| Format | Topology | Workload | C | Runs | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["grouped"]:
        lines.append(f"| {r['format']} | {r['topology']} | {r['workload']} | {r['concurrency']} | {r['runs']} | {r['ttft_p95_ms']:.2f} | {r['tpot_p95_ms']:.2f} | {r['e2e_p95_ms']:.2f} |")
    lines += ["", "## Overlapping BF16/FP8 configurations", "", "| Topology | Workload | C | FP8 E2E p95 delta | FP8 TTFT p95 delta |", "|---|---|---:|---:|---:|"]
    for r in summary["overlap"]:
        lines.append(f"| {r['topology']} | {r['workload']} | {r['concurrency']} | {r['e2e_p95_delta_pct_fp8_vs_bf16']:+.2f}% | {r['ttft_p95_delta_pct_fp8_vs_bf16']:+.2f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(load(args.inputs))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"rows": result["row_count"], "completed": result["completed_row_count"], "groups": len(result["grouped"])}, indent=2))


if __name__ == "__main__":
    main()
