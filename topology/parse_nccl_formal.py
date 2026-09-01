#!/usr/bin/env python3
# 作用：解析 nccl-tests 日志并生成统计表。
"""Parse raw nccl-tests logs into summarized statistics.

Reads ``all_reduce/all_gather/reduce_scatter/alltoall/sendrecv_*_r[1-5].log``
from a formal run directory, computes mean/CI95 per (collective, label, size)
and writes ``statistics.json``/``statistics.csv`` for
``scripts/build_nccl_cost_db.py``.
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

NAME_RE = re.compile(
    r"^(all_reduce|sendrecv|all_gather|reduce_scatter|alltoall)_(.+)_r([1-5])\.log$"
)
T_CRIT_95_DF4 = 2.7764451051977987


def parse_rows(path: Path):
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 13 or not fields[0].isdigit():
            continue
        try:
            rows[int(fields[0])] = {
                "time_us": float(fields[5]),
                "algbw_gbps": float(fields[6]),
                "busbw_gbps": float(fields[7]),
                "wrong": int(fields[8]),
            }
        except ValueError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    root = Path(args.run_dir)
    logs = root / "logs"

    records = []
    for path in sorted(logs.glob("*.log")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        collective, label, repetition = match.groups()
        for size, metrics in parse_rows(path).items():
            records.append({
                "collective": collective,
                "label": label,
                "repetition": int(repetition),
                "size_bytes": size,
                **metrics,
            })

    if not records:
        raise RuntimeError(f"no nccl-tests rows parsed from {logs}")

    groups = {}
    for row in records:
        key = (row["collective"], row["label"], row["size_bytes"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (collective, label, size), rows in sorted(groups.items()):
        for metric in ("time_us", "algbw_gbps", "busbw_gbps"):
            values = [row[metric] for row in rows]
            n = len(values)
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if n > 1 else 0.0
            ci95 = T_CRIT_95_DF4 * stdev / math.sqrt(n) if n == 5 else None
            summary.append({
                "collective": collective,
                "label": label,
                "size_bytes": size,
                "metric": metric,
                "n": n,
                "mean": mean,
                "sample_stdev": stdev,
                "ci95_half_width": ci95,
                "min": min(values),
                "max": max(values),
                "wrong_total": sum(row["wrong"] for row in rows),
            })

    csv_path = root / "statistics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (root / "statistics.json").write_text(json.dumps(summary, indent=2))

    selected_sizes = {4096, 65536, 1048576, 67108864, 268435456}
    selected = [r for r in summary
                if r["size_bytes"] in selected_sizes
                and r["metric"] in {"time_us", "algbw_gbps"}]
    (root / "selected-statistics.json").write_text(json.dumps(selected, indent=2))

    result = {
        "log_files": len({(r["collective"], r["label"], r["repetition"])
                          for r in records}),
        "parsed_measurements": len(records),
        "statistics_rows": len(summary),
        "all_groups_n5": all(r["n"] == 5 for r in summary),
        "wrong_total": sum(r["wrong"] for r in records),
        "outputs": [str(csv_path), str(root / "statistics.json"),
                    str(root / "selected-statistics.json")],
    }
    (root / "parse-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["all_groups_n5"] and result["wrong_total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
