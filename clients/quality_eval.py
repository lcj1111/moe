#!/usr/bin/env python3
"""Run deterministic, non-code-executing quality evaluation through an API."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import statistics
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any


def parse_choice(text: str) -> str | None:
    patterns = [
        r"(?:FINAL|Answer|答案)\s*[:：]\s*\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?(?:\s|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\b([A-J])\b", text.upper())
    return matches[-1] if matches else None


def normalize_number(value: str) -> Decimal | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".。")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_number(text: str) -> Decimal | None:
    matches = re.findall(
        r"(?:FINAL|Answer|答案)\s*[:：]\s*([-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?)",
        text, flags=re.IGNORECASE,
    )
    if not matches:
        matches = re.findall(r"[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    return normalize_number(matches[-1]) if matches else None


def score(row: dict[str, Any], text: str) -> tuple[bool | None, str | None]:
    if row["score_type"] == "multiple_choice":
        prediction = parse_choice(text)
        return prediction == row["answer"], prediction
    if row["score_type"] == "numeric":
        prediction_number = parse_number(text)
        expected_number = normalize_number(str(row["answer"]))
        prediction = str(prediction_number) if prediction_number is not None else None
        return prediction_number is not None and prediction_number == expected_number, prediction
    if row["score_type"] == "deferred_code":
        return None, None
    raise ValueError(f"unsupported score_type={row['score_type']}")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


async def run(args: argparse.Namespace) -> int:
    import aiohttp

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if not args.include_deferred_code:
        rows = [row for row in rows if row["score_type"] != "deferred_code"]
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(index: int, row: dict[str, Any]) -> None:
            payload = {
                "model": args.model, "messages": row["messages"],
                "temperature": 0, "seed": args.seed,
                "max_tokens": int(row["max_tokens"]), "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.perf_counter()
            error = None
            response_json = None
            text = ""
            try:
                async with semaphore:
                    async with session.post(
                        args.base_url.rstrip("/") + "/chat/completions", json=payload
                    ) as response:
                        body = await response.text()
                        if response.status != 200:
                            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
                        response_json = json.loads(body)
                text = response_json["choices"][0]["message"]["content"]
            except Exception as exc:  # retained in artifact and reflected in exit code
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - started) * 1000
            correct, prediction = (None, None) if error else score(row, text)
            output_rows[index] = {
                "id": row["id"], "benchmark": row["benchmark"],
                "score_type": row["score_type"], "expected": row.get("answer"),
                "prediction": prediction, "correct": correct, "response": text,
                "elapsed_ms": elapsed_ms, "error": error,
                "usage": (response_json or {}).get("usage"),
            }

        await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))

    results = [row for row in output_rows if row is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_benchmark[row["benchmark"]].append(row)
    summary = {
        "base_url": args.base_url, "model": args.model,
        "manifest": str(args.manifest), "seed": args.seed,
        "requested": len(rows), "completed": sum(not row["error"] for row in results),
        "failed": sum(bool(row["error"]) for row in results), "benchmarks": {},
    }
    for benchmark, items in sorted(by_benchmark.items()):
        scored = [row for row in items if row["correct"] is not None]
        latencies = [row["elapsed_ms"] for row in items if not row["error"]]
        summary["benchmarks"][benchmark] = {
            "records": len(items), "scored": len(scored),
            "correct": sum(bool(row["correct"]) for row in scored),
            "accuracy": (sum(bool(row["correct"]) for row in scored) / len(scored))
            if scored else None,
            "latency_ms": {"mean": statistics.fmean(latencies) if latencies else None,
                           "p50": percentile(latencies, 0.50),
                           "p95": percentile(latencies, 0.95)},
        }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["failed"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--include-deferred-code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
