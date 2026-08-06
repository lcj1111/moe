"""Slice a small timing pilot from the frozen full-set official manifest.

The full set (MMLU-Pro test + C-Eval test) is ~24,374 requests and each may
take tens of seconds (thinking mode), so a pilot of 16 + 16 rows is used to
measure single-request latency before committing to a multi-hour full run.

Usage (server):
    /data/models/test/qtopomoe_quant_env/bin/python evaluation/slice_pilot.py \
        --src /data/models/test/qtopomoe_quality/full_set_official_v1.jsonl \
        --out /data/models/test/qtopomoe_quality/full_set_pilot_v1.jsonl \
        --mmlu 16 --ceval 16
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=pathlib.Path, default="/data/models/test/qtopomoe_quality/full_set_official_v1.jsonl")
    parser.add_argument("--out", type=pathlib.Path, default="/data/models/test/qtopomoe_quality/full_set_pilot_v1.jsonl")
    parser.add_argument("--mmlu", type=int, default=16)
    parser.add_argument("--ceval", type=int, default=16)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mmlu = [r for r in rows if r["benchmark"] == "mmlu_pro"][: args.mmlu]
    ceval = [r for r in rows if r["benchmark"] == "ceval"][: args.ceval]
    with args.out.open("w", encoding="utf-8") as handle:
        for row in mmlu + ceval:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        "PILOT_ROWS",
        len(mmlu) + len(ceval),
        "MMLU",
        len(mmlu),
        "CEVAL",
        len(ceval),
        "->",
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
