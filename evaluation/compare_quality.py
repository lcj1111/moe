#!/usr/bin/env python3
"""Compare a quantized quality summary with its BF16 reference."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-drop-points", type=float, default=2.0)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    rows = {}
    gate_pass = reference.get("failed") == 0 and candidate.get("failed") == 0
    for benchmark, ref in sorted(reference["benchmarks"].items()):
        if benchmark not in candidate["benchmarks"]:
            raise SystemExit(f"candidate is missing benchmark {benchmark}")
        cand = candidate["benchmarks"][benchmark]
        ref_accuracy = ref["accuracy"]
        cand_accuracy = cand["accuracy"]
        if ref["records"] != cand["records"] or ref["scored"] != cand["scored"]:
            raise SystemExit(f"record mismatch for {benchmark}")
        drop_points = (ref_accuracy - cand_accuracy) * 100
        passed = drop_points <= args.max_drop_points
        gate_pass = gate_pass and passed
        rows[benchmark] = {
            "records": ref["records"], "reference_accuracy": ref_accuracy,
            "candidate_accuracy": cand_accuracy, "drop_points": drop_points,
            "threshold_points": args.max_drop_points, "pass": passed,
        }
    payload = {
        "schema_version": "qtopomoe.quality_comparison.v1",
        "reference_model": reference["model"], "candidate_model": candidate["model"],
        "manifest": reference["manifest"], "benchmarks": rows,
        "gate_pass": gate_pass,
        "note": "HumanEval is excluded until isolated code execution is available.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
