#!/usr/bin/env python3
# 作用：确定性地把实测 kernel 行合并进版本化数据库。
"""Merge measured kernel rows into the Phase 4 database deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("backend"),
        row.get("precision"),
        int(row.get("m_bucket", -1)),
        row.get("kernel_name"),
        row.get("source"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    database = json.loads(args.database.read_text(encoding="utf-8"))
    incoming = json.loads(args.measurements.read_text(encoding="utf-8"))
    if database.get("schema_version") != "qtopomoe.kernel_db.v1":
        raise SystemExit("unsupported kernel database schema")
    if not isinstance(incoming, list):
        raise SystemExit("measurement input must be a JSON list")
    for row in incoming:
        if row.get("schema_version") != "qtopomoe.kernel_measurement.v1":
            raise SystemExit("unsupported measurement schema")
        if not row.get("measured") or not row.get("valid"):
            raise SystemExit(f"refusing invalid measurement: {row_key(row)}")

    rows = list(database["measurements"])
    positions = {row_key(row): index for index, row in enumerate(rows)}
    before = len(rows)
    for row in incoming:
        key = row_key(row)
        if key in positions:
            rows[positions[key]] = row
        else:
            positions[key] = len(rows)
            rows.append(row)
    output = args.output or args.database
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"measurements": rows,
                    "schema_version": "qtopomoe.kernel_db.v1"},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"before": before, "incoming": len(incoming),
                      "after": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
