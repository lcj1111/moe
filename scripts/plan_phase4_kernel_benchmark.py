#!/usr/bin/env python3
"""Create a reproducible Phase 4 kernel benchmark plan.

This command is plan-only by default.  It never allocates a CUDA tensor or
starts a process unless an explicit execution wrapper is supplied later.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_plan(manifest: dict, backends: list[str], precisions: list[str], command_template: str | None) -> dict:
    buckets = sorted({int(r["prefill_m_bucket"]) for r in manifest["records"]} |
                     {int(r["decode_m_bucket"]) for r in manifest["records"]})
    rows = []
    for precision in precisions:
        for backend in backends:
            for m_bucket in buckets:
                command = None
                if command_template:
                    command = command_template.format(backend=backend, precision=precision, m_bucket=m_bucket)
                rows.append({"backend": backend, "precision": precision, "m_bucket": m_bucket,
                             "command": command, "status": "planned", "measured": False,
                             "result_schema": "qtopomoe.kernel_measurement.v1"})
    return {"schema_version": "qtopomoe.kernel_benchmark_plan.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": manifest.get("sha256"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-buckets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="cutlass,triton,flashinfer")
    parser.add_argument("--precisions", default="bf16,fp8")
    parser.add_argument("--command-template", help="optional command template; not executed by this script")
    args = parser.parse_args()
    manifest = json.loads(args.m_buckets.read_text(encoding="utf-8"))
    plan = build_plan(manifest, [x for x in args.backends.split(",") if x],
                      [x for x in args.precisions.split(",") if x], args.command_template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned {len(plan['rows'])} kernel runs; no CUDA command was executed")


if __name__ == "__main__":
    main()
