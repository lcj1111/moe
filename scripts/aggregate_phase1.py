#!/usr/bin/env python3
# 作用：汇总 Phase 1 各拓扑服务结果并生成统一矩阵。
"""Aggregate Phase 1 service-matrix smoke summaries into one table.

Reads per-topology ``*_c*.summary.json`` files under a run root and writes a
single CSV/JSON matrix (format, topology, workload, concurrency, TTFT/TPOT/
e2e).  Companion to ``scripts/analyze_phase1.py`` and
``scripts/phase1_matrix.sh``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.run_root)
    rows = []
    for topology in sorted(p for p in root.iterdir() if p.is_dir()):
        acceptance = topology / "client/acceptance.json"
        for summary_path in sorted((topology / "client").glob("*.summary.json")):
            summary = json.loads(summary_path.read_text())
            rows.append({
                "format": root.name,
                "topology": topology.name,
                "acceptance": acceptance.exists(),
                "workload": "short" if summary["input_tokens_requested"] == 256 else "medium",
                "input_tokens": summary["input_tokens_requested"],
                "output_tokens": summary["output_tokens_requested"],
                "concurrency": summary["concurrency"],
                "requests": summary["requests"],
                "completed": summary["completed"],
                "failed": summary["failed"],
                "ttft_p50_ms": summary["ttft_ms"]["p50"],
                "ttft_p95_ms": summary["ttft_ms"]["p95"],
                "tpot_p50_ms": summary["tpot_ms"]["p50"],
                "tpot_p95_ms": summary["tpot_ms"]["p95"],
                "e2e_p50_ms": summary["e2e_ms"]["p50"],
                "e2e_p95_ms": summary["e2e_ms"]["p95"],
                "summary_path": str(summary_path),
            })
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["format", "topology"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "json": str(out), "failures": sum(r["failed"] for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
