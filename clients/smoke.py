#!/usr/bin/env python3
"""Deterministic OpenAI-compatible concurrency smoke with TTFT/TPOT estimates."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def prompt_for_tokens(n: int) -> str:
    # The server tokenizer is authoritative; this deterministic text keeps the
    # requested workload shape stable without requiring tokenizer imports.
    return " ".join(f"token{i % 1000:04d}" for i in range(max(1, n)))


def one_request(base_url: str, model: str, input_tokens: int, output_tokens: int,
                request_id: int, seed: int, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for_tokens(input_tokens)}],
        "temperature": 0,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed + request_id,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    first = None
    text_parts: list[str] = []
    usage = None
    status = "ok"
    error = None
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if first is None:
                    first = time.perf_counter()
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {}) or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status = "error"
        error = repr(exc)
    finished = time.perf_counter()
    ttft_ms = (first - started) * 1000 if first else None
    e2e_ms = (finished - started) * 1000
    generated = int((usage or {}).get("completion_tokens") or max(1, round(len("".join(text_parts)) / 4)))
    tpot_ms = ((e2e_ms - (ttft_ms or 0)) / max(1, generated - 1)) if ttft_ms is not None else None
    return {
        "request_id": request_id,
        "status": status,
        "error": error,
        "input_tokens_requested": input_tokens,
        "output_tokens_requested": output_tokens,
        "output_tokens": generated,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "seed": seed + request_id,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-tokens", type=int, required=True)
    ap.add_argument("--output-tokens", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        ap.error("concurrency and requests must be positive")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, args.base_url, args.model,
                               args.input_tokens, args.output_tokens, i,
                               args.seed, args.timeout)
                   for i in range(args.requests)]
        records = [f.result() for f in futures]
    records.sort(key=lambda x: x["request_id"])
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    good = [x for x in records if x["status"] == "ok"]
    ttft = [x["ttft_ms"] for x in good if x["ttft_ms"] is not None]
    tpot = [x["tpot_ms"] for x in good if x["tpot_ms"] is not None]
    e2e = [x["e2e_ms"] for x in good]
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "input_tokens_requested": args.input_tokens,
        "output_tokens_requested": args.output_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "completed": len(good),
        "failed": len(records) - len(good),
        "ttft_ms": {"p50": percentile(ttft, .50), "p95": percentile(ttft, .95), "mean": statistics.mean(ttft) if ttft else None},
        "tpot_ms": {"p50": percentile(tpot, .50), "p95": percentile(tpot, .95), "mean": statistics.mean(tpot) if tpot else None},
        "e2e_ms": {"p50": percentile(e2e, .50), "p95": percentile(e2e, .95), "mean": statistics.mean(e2e) if e2e else None},
        "output_tokens_total": sum(x["output_tokens"] for x in good),
        "wall_time_s": (max(e2e) / 1000) if e2e else None,
    }
    summary_path = Path(args.summary) if args.summary else out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
