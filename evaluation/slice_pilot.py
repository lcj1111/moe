"""Slice a stratified timing pilot from a frozen full-set manifest.

The full set (MMLU-Pro test + C-Eval test) is ~24,374 requests; even in
benchmark-official (no-thinking) mode each request can take seconds, so a
small stratified pilot is used to measure single-request latency before
committing to a full run.

Sampling is stratified by category (MMLU-Pro) and subject (C-Eval) so the
pilot is not dominated by one contiguous slice of the file.

Usage (server):
    /data/models/test/qtopomoe_quant_env/bin/python evaluation/slice_pilot.py \
        --src /data/models/test/qtopomoe_quality/full_set_official_protocol_v1.jsonl \
        --out /data/models/test/qtopomoe_quality/full_set_official_protocol_pilot_v1.jsonl \
        --mmlu 4 --ceval 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import defaultdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=pathlib.Path, default="/data/models/test/qtopomoe_quality/full_set_official_v1.jsonl")
    parser.add_argument("--out", type=pathlib.Path, default="/data/models/test/qtopomoe_quality/full_set_pilot_v1.jsonl")
    parser.add_argument("--mmlu", type=int, default=16)
    parser.add_argument("--ceval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Split on "\n" only; splitlines() would treat Unicode NEL (\x85) inside
    # prompt text as a row boundary and truncate the JSON.
    rows = [
        json.loads(line)
        for line in args.src.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    def stratified(items: list[dict], key: str, count: int) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in items:
            buckets[str(row[key])].append(row)
        order = sorted(buckets)
        picked: list[dict] = []
        rng = random.Random(args.seed)
        if count >= len(order):
            # draw one per bucket, then fill round-robin
            for name in order:
                picked.append(rng.choice(buckets[name]))
            remaining = count - len(order)
            for _ in range(remaining):
                picked.append(rng.choice(buckets[rng.choice(order)]))
        else:
            chosen_buckets = rng.sample(order, count)
            for name in chosen_buckets:
                picked.append(rng.choice(buckets[name]))
        return picked

    mmlu = stratified(
        [r for r in rows if r["benchmark"] == "mmlu_pro"], "category", args.mmlu
    )
    ceval = stratified(
        [r for r in rows if r["benchmark"] == "ceval"], "subject", args.ceval
    )
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
