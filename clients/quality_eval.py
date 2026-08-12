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
        r'"answer"\s*:\s*"?([A-J])"?',
        r"[Tt]he answer is\s*\(?([A-J])\)?",
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

    # Split on "\n" only: splitlines() also treats Unicode NEL (\x85) and
    # other line separators as boundaries, which can truncate a JSON row.
    rows = [json.loads(line) for line in args.manifest.read_text().split("\n") if line]
    if not args.include_deferred_code:
        rows = [row for row in rows if row["score_type"] != "deferred_code"]
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    row_indexes = {row["id"]: index for index, row in enumerate(rows)}
    if len(row_indexes) != len(rows):
        raise ValueError("manifest contains duplicate ids")
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    resumed = 0
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            previous = json.loads(line)
            index = row_indexes.get(previous.get("id"))
            if index is not None and not previous.get("error"):
                output_rows[index] = previous
                resumed += 1
    checkpoint_lock = asyncio.Lock()
    completed_since_checkpoint = 0

    def write_checkpoint() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for item in output_rows:
                if item is not None:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(index: int, row: dict[str, Any]) -> None:
            nonlocal completed_since_checkpoint
            sampling = row.get("sampling", {})
            payload = {
                "model": args.model, "messages": row["messages"],
                "temperature": sampling.get("temperature", 0),
                "seed": sampling.get("seed", args.seed),
                "max_tokens": int(row["max_tokens"]), "stream": False,
                "chat_template_kwargs": {
                    "enable_thinking": sampling.get("enable_thinking", False)
                },
            }
            for key in (
                "top_p", "top_k", "min_p", "presence_penalty",
                "frequency_penalty", "repetition_penalty",
            ):
                if key in sampling:
                    payload[key] = sampling[key]
            queued_at = time.perf_counter()
            request_started: float | None = None
            error = None
            response_json = None
            text = ""
            try:
                async with semaphore:
                    request_started = time.perf_counter()
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
            finished_at = time.perf_counter()
            queue_ms = ((request_started or finished_at) - queued_at) * 1000
            request_ms = (
                (finished_at - request_started) * 1000
                if request_started is not None else None
            )
            elapsed_ms = (finished_at - queued_at) * 1000
            choice = ((response_json or {}).get("choices") or [{}])[0]
            usage = (response_json or {}).get("usage") or {}
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
            finish_reason = choice.get("finish_reason")
            truncated = finish_reason == "length" or (
                completion_tokens is not None
                and int(completion_tokens) >= int(row["max_tokens"])
            )
            if error:
                correct, prediction = None, None
            else:
                correct, prediction = score(row, text)
                if truncated:
                    correct = None
            output_rows[index] = {
                "id": row["id"], "benchmark": row["benchmark"],
                "score_type": row["score_type"], "expected": row.get("answer"),
                "prediction": prediction, "correct": correct, "response": text,
                "queue_ms": queue_ms, "request_ms": request_ms,
                "elapsed_ms": elapsed_ms, "finish_reason": finish_reason,
                "max_tokens": int(row["max_tokens"]), "truncated": truncated,
                "error": error, "usage": usage,
            }

            async with checkpoint_lock:
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= args.checkpoint_every:
                    write_checkpoint()
                    completed_since_checkpoint = 0

        await asyncio.gather(*(
            one(i, row) for i, row in enumerate(rows) if output_rows[i] is None
        ))

    results = [row for row in output_rows if row is not None]
    write_checkpoint()
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_benchmark[row["benchmark"]].append(row)
    summary = {
        "base_url": args.base_url, "model": args.model,
        "manifest": str(args.manifest), "seed": args.seed,
        "protocols": sorted({row.get("protocol", "regression_v1") for row in rows}),
        "requested": len(rows), "completed": sum(not row["error"] for row in results),
        "resumed": resumed,
        "failed": sum(bool(row["error"]) for row in results),
        "truncated": sum(bool(row["truncated"]) for row in results),
        "benchmarks": {},
    }
    for benchmark, items in sorted(by_benchmark.items()):
        scored = [row for row in items if row["correct"] is not None]
        request_latencies = [
            row["request_ms"] for row in items
            if not row["error"] and row["request_ms"] is not None
        ]
        queue_times = [row["queue_ms"] for row in items if not row["error"]]
        end_to_end = [row["elapsed_ms"] for row in items if not row["error"]]
        def timing(values: list[float]) -> dict[str, float | None]:
            return {
                "mean": statistics.fmean(values) if values else None,
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
            }
        summary["benchmarks"][benchmark] = {
            "records": len(items), "scored": len(scored),
            "truncated": sum(bool(row["truncated"]) for row in items),
            "correct": sum(bool(row["correct"]) for row in scored),
            "accuracy": (sum(bool(row["correct"]) for row in scored) / len(scored))
            if scored else None,
            "request_latency_ms": timing(request_latencies),
            "queue_ms": timing(queue_times),
            "end_to_end_ms": timing(end_to_end),
        }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["failed"] or summary["truncated"] else 0


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--include-deferred-code", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
