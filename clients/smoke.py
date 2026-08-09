#!/usr/bin/env python3
"""Deterministic OpenAI-compatible concurrency smoke with TTFT/TPOT estimates."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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


def exact_chat_prompt(tokenizer_path: str, target_tokens: int) -> tuple[str, int, str]:
    """Build user content whose rendered chat prompt has exactly target_tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def rendered_tokens(content: str) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(input_ids)

    base = rendered_tokens("")
    if target_tokens < base:
        raise ValueError(f"target {target_tokens} is below chat-template minimum {base}")
    # For the frozen Qwen tokenizer, each leading-space ``x`` adds one token.
    # Keep a bounded fallback search so a tokenizer/template change fails
    # explicitly instead of silently changing the workload.
    count = target_tokens - base
    content = " x" * count
    actual = rendered_tokens(content)
    if actual != target_tokens:
        lo, hi = 0, max(target_tokens * 2, 32)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " x" * mid
            measured = rendered_tokens(candidate)
            if measured == target_tokens:
                content, actual = candidate, measured
                break
            if measured < target_tokens:
                lo = mid + 1
            else:
                hi = mid - 1
        else:
            raise ValueError(
                f"cannot construct exact {target_tokens}-token prompt; nearest search ended at {actual}"
            )
    template = tokenizer.chat_template or ""
    return content, actual, hashlib.sha256(template.encode("utf-8")).hexdigest()


def one_request(base_url: str, model: str, input_tokens: int, output_tokens: int,
                request_id: int, seed: int, timeout: float,
                prompt_text: str | None = None,
                input_tokens_actual: int | None = None) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": (
            prompt_text if prompt_text is not None else prompt_for_tokens(input_tokens))}],
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
        "input_tokens_actual": input_tokens_actual,
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
    ap.add_argument("--tokenizer", default=None,
                    help="checkpoint/tokenizer path; makes rendered chat input token-exact")
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        ap.error("concurrency and requests must be positive")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt_text = None
    input_tokens_actual = None
    chat_template_sha256 = None
    if args.tokenizer:
        prompt_text, input_tokens_actual, chat_template_sha256 = exact_chat_prompt(
            args.tokenizer, args.input_tokens)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, args.base_url, args.model,
                               args.input_tokens, args.output_tokens, i,
                               args.seed, args.timeout, prompt_text,
                               input_tokens_actual)
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
        "input_tokens_actual": input_tokens_actual,
        "tokenizer": args.tokenizer,
        "chat_template_sha256": chat_template_sha256,
        "prompt_sha256": (hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                           if prompt_text is not None else None),
        "output_tokens_requested": args.output_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "completed": len(good),
        "failed": len(records) - len(good),
        "ttft_ms": {"p50": percentile(ttft, .50), "p95": percentile(ttft, .95), "p99": percentile(ttft, .99), "mean": statistics.mean(ttft) if ttft else None},
        "tpot_ms": {"p50": percentile(tpot, .50), "p95": percentile(tpot, .95), "p99": percentile(tpot, .99), "mean": statistics.mean(tpot) if tpot else None},
        "e2e_ms": {"p50": percentile(e2e, .50), "p95": percentile(e2e, .95), "p99": percentile(e2e, .99), "mean": statistics.mean(e2e) if e2e else None},
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
