#!/usr/bin/env python3
"""Reproducible OpenAI-compatible service workload and cache audit client."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def prompt_for_tokens(n: int, request_id: int = 0) -> str:
    """Return deterministic, request-distinct text when no tokenizer is supplied."""
    marker = f"request{request_id:06d}"
    return marker + " " + " ".join(f"token{i % 1000:04d}" for i in range(max(1, n - 1)))


def rendered_ids(tokenizer: Any, content: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    values = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return list(values)


def common_prefix_tokens(rows: list[list[int]]) -> int:
    if not rows:
        return 0
    limit = min(len(row) for row in rows)
    for index in range(limit):
        value = rows[0][index]
        if any(row[index] != value for row in rows[1:]):
            return index
    return limit


def _exact_content(tokenizer: Any, target_tokens: int, stem: str) -> tuple[str, list[int]]:
    """Append a stable one-token filler until the rendered chat length is exact."""
    stem_ids = rendered_ids(tokenizer, stem)
    if len(stem_ids) > target_tokens:
        raise ValueError(f"stem already renders to {len(stem_ids)} > {target_tokens}")
    lo, hi = 0, max(target_tokens * 2, 32)
    while lo <= hi:
        count = (lo + hi) // 2
        content = stem + " x" * count
        ids = rendered_ids(tokenizer, content)
        if len(ids) == target_tokens:
            return content, ids
        if len(ids) < target_tokens:
            lo = count + 1
        else:
            hi = count - 1
    raise ValueError(f"cannot construct an exact {target_tokens}-token prompt from stem")


def _unique_fragments(tokenizer: Any, count: int) -> list[str]:
    """Find printable, single-token fragments with distinct token ids."""
    fragments: list[str] = []
    for token_id in range(len(tokenizer)):
        try:
            fragment = tokenizer.decode([token_id], skip_special_tokens=True)
            encoded = tokenizer.encode(fragment, add_special_tokens=False)
        except (TypeError, ValueError, UnicodeError):
            continue
        if (not fragment or len(fragment) > 24 or "\n" in fragment or "\r" in fragment
                or not fragment.isprintable() or encoded != [token_id]
                or not fragment.strip()):
            continue
        fragments.append(fragment)
        if len(fragments) >= count:
            return fragments
    raise ValueError(f"tokenizer exposes only {len(fragments)} usable unique fragments; need {count}")


def build_prompt_plan(tokenizer_path: str, target_tokens: int, requests: int,
                      prefix_cache_pct: int, seed: int) -> dict[str, Any]:
    """Build exact-length prompts plus an excluded cache-prewarm prompt."""
    from transformers import AutoTokenizer

    if prefix_cache_pct not in (0, 50, 100):
        raise ValueError("prefix_cache_pct must be one of 0, 50, 100")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    template = tokenizer.chat_template or ""
    empty_ids = rendered_ids(tokenizer, "")
    if target_tokens < len(empty_ids):
        raise ValueError(
            f"target {target_tokens} is below chat-template minimum {len(empty_ids)}")

    if prefix_cache_pct == 100:
        content, ids = _exact_content(tokenizer, target_tokens, "")
        contents = [content] * requests
        token_rows = [ids] * requests
        prewarm_content = content
    else:
        fragments = _unique_fragments(tokenizer, requests + 1)
        fixed_prefix = common_prefix_tokens([
            rendered_ids(tokenizer, fragments[0]),
            rendered_ids(tokenizer, fragments[1]),
        ]) if len(fragments) > 1 else len(empty_ids)
        desired_lcp = round(target_tokens * prefix_cache_pct / 100)
        common_fill = max(0, desired_lcp - fixed_prefix)
        common_stem = " x" * common_fill
        contents, token_rows = [], []
        for fragment in fragments[:requests]:
            content, ids = _exact_content(tokenizer, target_tokens, common_stem + fragment)
            contents.append(content)
            token_rows.append(ids)
        prewarm_content, _ = _exact_content(
            tokenizer, target_tokens, common_stem + fragments[-1])

    rendered_lcp = common_prefix_tokens(token_rows)
    salt_root = hashlib.sha256(
        f"qtopomoe:{tokenizer_path}:{target_tokens}:{prefix_cache_pct}:{seed}".encode()
    ).hexdigest()[:32]
    prompts = []
    for request_id, (content, ids) in enumerate(zip(contents, token_rows)):
        cache_salt = (f"{salt_root}-{request_id:08d}"
                      if prefix_cache_pct == 0 else salt_root)
        prompts.append({
            "request_id": request_id,
            "text": content,
            "prompt_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "cache_salt": cache_salt,
            "input_tokens_actual": len(ids),
        })
    return {
        "prompts": prompts,
        "prewarm": {
            "text": prewarm_content,
            "cache_salt": salt_root,
            "input_tokens_actual": target_tokens,
        } if prefix_cache_pct else None,
        "input_tokens_actual": target_tokens,
        "prefix_cache_pct_target": prefix_cache_pct,
        "rendered_common_prefix_tokens": rendered_lcp,
        "rendered_common_prefix_pct": 100 * rendered_lcp / target_tokens,
        "unique_prompt_count": len({row["prompt_sha256"] for row in prompts}),
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def arrival_offsets(mode: str, requests: int, concurrency: int,
                    request_rate: float | None, seed: int) -> list[float]:
    """Return deterministic open-loop enqueue offsets; closed-loop is dynamic."""
    if mode == "closed_loop":
        return [0.0] * requests
    if request_rate is None or not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("positive --request-rate is required for poisson and burst")
    if mode == "poisson":
        rng = random.Random(seed)
        offsets, current = [0.0], 0.0
        for _ in range(1, requests):
            current += rng.expovariate(request_rate)
            offsets.append(current)
        return offsets
    if mode == "burst":
        interval = concurrency / request_rate
        return [(request_id // concurrency) * interval for request_id in range(requests)]
    raise ValueError(f"unknown arrival mode: {mode}")


def cached_tokens_from_usage(usage: dict[str, Any] | None) -> int | None:
    details = (usage or {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if value is not None else None


def one_request(base_url: str, model: str, input_tokens: int, output_tokens: int,
                request_id: int, seed: int, timeout: float,
                prompt_text: str | None = None,
                input_tokens_actual: int | None = None,
                cache_salt: str | None = None,
                prompt_sha256: str | None = None,
                run_epoch: float | None = None,
                scheduled_offset_s: float | None = None,
                submitted_offset_s: float | None = None) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": (
            prompt_text if prompt_text is not None else prompt_for_tokens(input_tokens, request_id))}],
        "temperature": 0,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed + request_id,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    started = time.perf_counter()
    first = None
    text_parts: list[str] = []
    usage = None
    finish_reason = None
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
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {}) or {}
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        text_parts.append(content)
    except urllib.error.HTTPError as exc:
        status = "error"
        try:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
        except OSError:
            body = ""
        error = f"HTTPError({exc.code}): {body}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status = "error"
        error = repr(exc)
    finished = time.perf_counter()
    ttft_ms = (first - started) * 1000 if first else None
    e2e_ms = (finished - started) * 1000
    generated = int((usage or {}).get("completion_tokens")
                    or max(1, round(len("".join(text_parts)) / 4)))
    prompt_tokens = (usage or {}).get("prompt_tokens")
    cached_tokens = cached_tokens_from_usage(usage)
    tpot_ms = ((e2e_ms - (ttft_ms or 0)) / max(1, generated - 1)
               if ttft_ms is not None else None)
    epoch = run_epoch if run_epoch is not None else started
    return {
        "request_id": request_id,
        "status": status,
        "error": error,
        "finish_reason": finish_reason,
        "input_tokens_requested": input_tokens,
        "input_tokens_actual": input_tokens_actual,
        "prompt_tokens_reported": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cached_token_ratio": (cached_tokens / prompt_tokens
                               if cached_tokens is not None and prompt_tokens else None),
        "prompt_sha256": prompt_sha256,
        "cache_salt_sha256": (hashlib.sha256(cache_salt.encode()).hexdigest()
                              if cache_salt is not None else None),
        "output_tokens_requested": output_tokens,
        "output_tokens": generated,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "scheduled_offset_s": scheduled_offset_s,
        "submitted_offset_s": submitted_offset_s,
        "started_offset_s": started - epoch,
        "finished_offset_s": finished - epoch,
        "seed": seed + request_id,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[index]


def metric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, .50), "p95": percentile(values, .95),
        "p99": percentile(values, .99),
        "mean": statistics.mean(values) if values else None,
    }


def peak_in_flight(records: list[dict[str, Any]]) -> int:
    events = []
    for row in records:
        events.extend([(row["started_offset_s"], 1), (row["finished_offset_s"], -1)])
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def execute_requests(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epoch = time.perf_counter()
    common = (args.base_url, args.model, args.input_tokens, args.output_tokens,
              args.seed, args.timeout)

    def submit(pool: concurrent.futures.ThreadPoolExecutor, request_id: int,
               scheduled: float) -> concurrent.futures.Future[dict[str, Any]]:
        prompt = prompts[request_id]
        submitted = time.perf_counter() - epoch
        return pool.submit(
            one_request, common[0], common[1], common[2], common[3], request_id,
            common[4], common[5], prompt.get("text"), prompt.get("input_tokens_actual"),
            prompt.get("cache_salt"), prompt.get("prompt_sha256"), epoch,
            scheduled, submitted)

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.arrival_mode == "closed_loop":
            pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
            next_id = 0
            while next_id < min(args.concurrency, args.requests):
                pending.add(submit(pool, next_id, 0.0))
                next_id += 1
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                records.extend(future.result() for future in done)
                while next_id < args.requests and len(pending) < args.concurrency:
                    pending.add(submit(pool, next_id, 0.0))
                    next_id += 1
        else:
            offsets = arrival_offsets(args.arrival_mode, args.requests, args.concurrency,
                                      args.request_rate, args.stream_seed)
            futures = []
            for request_id, scheduled in enumerate(offsets):
                remaining = epoch + scheduled - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                futures.append(submit(pool, request_id, scheduled))
            records = [future.result() for future in futures]
    return sorted(records, key=lambda row: row["request_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--tokenizer", default=None,
                        help="checkpoint/tokenizer path; makes chat input token-exact")
    parser.add_argument("--prefix-cache-pct", type=int, choices=(0, 50, 100), default=0)
    parser.add_argument("--arrival-mode", choices=("closed_loop", "poisson", "burst"),
                        default="closed_loop")
    parser.add_argument("--request-rate", type=float, default=None,
                        help="offered requests/s; required for poisson and burst")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--require-cache-details", action="store_true")
    parser.add_argument("--cache-ratio-tolerance", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")
    if args.arrival_mode != "closed_loop" and (args.request_rate is None
                                                or args.request_rate <= 0):
        parser.error("positive --request-rate is required for poisson and burst")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.tokenizer:
        prompt_plan = build_prompt_plan(
            args.tokenizer, args.input_tokens, args.requests,
            args.prefix_cache_pct, args.seed)
    else:
        prompts = [{
            "request_id": request_id,
            "text": prompt_for_tokens(args.input_tokens, request_id),
            "prompt_sha256": None,
            "cache_salt": f"qtopomoe-no-tokenizer-{request_id}",
            "input_tokens_actual": None,
        } for request_id in range(args.requests)]
        prompt_plan = {
            "prompts": prompts, "prewarm": None, "input_tokens_actual": None,
            "prefix_cache_pct_target": args.prefix_cache_pct,
            "rendered_common_prefix_tokens": None,
            "rendered_common_prefix_pct": None,
            "unique_prompt_count": len(prompts), "chat_template_sha256": None,
        }

    prewarm_record = None
    if prompt_plan["prewarm"] and not args.no_prewarm:
        prewarm = prompt_plan["prewarm"]
        prewarm_record = one_request(
            args.base_url, args.model, args.input_tokens, 1, -1, args.seed,
            args.timeout, prewarm["text"], prewarm["input_tokens_actual"],
            prewarm["cache_salt"])
        if prewarm_record["status"] != "ok":
            raise RuntimeError(f"prefix cache prewarm failed: {prewarm_record['error']}")

    records = execute_requests(args, prompt_plan["prompts"])
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                      encoding="utf-8")
    good = [row for row in records if row["status"] == "ok"]
    ttft = [row["ttft_ms"] for row in good if row["ttft_ms"] is not None]
    tpot = [row["tpot_ms"] for row in good if row["tpot_ms"] is not None]
    e2e = [row["e2e_ms"] for row in good]
    cached_rows = [row for row in good if row["cached_tokens"] is not None]
    prompt_tokens_total = sum(int(row["prompt_tokens_reported"] or 0) for row in cached_rows)
    cached_tokens_total = sum(int(row["cached_tokens"] or 0) for row in cached_rows)
    actual_cache_ratio = (cached_tokens_total / prompt_tokens_total
                          if prompt_tokens_total else None)
    target_cache_ratio = args.prefix_cache_pct / 100
    tolerance = (args.cache_ratio_tolerance if args.cache_ratio_tolerance is not None
                 else max(0.15, 32 / args.input_tokens))
    cache_details_complete = len(cached_rows) == len(good)
    server_prompt_tokens_exact = (len(good) > 0 and all(
        row["prompt_tokens_reported"] == args.input_tokens for row in good))
    cache_ratio_gate = (actual_cache_ratio is not None
                        and abs(actual_cache_ratio - target_cache_ratio) <= tolerance)
    starts = sorted(row["started_offset_s"] for row in good)
    realized_interarrivals = [right - left for left, right in zip(starts, starts[1:])]
    scheduling_lag = [max(0.0, row["started_offset_s"] - row["scheduled_offset_s"])
                      for row in good if row["scheduled_offset_s"] is not None
                      and args.arrival_mode != "closed_loop"]
    wall_time = (max((row["finished_offset_s"] for row in good), default=0)
                 - min((row["started_offset_s"] for row in good), default=0))
    summary = {
        "schema_version": "qtopomoe.service_workload.v2",
        "base_url": args.base_url,
        "model": args.model,
        "input_tokens_requested": args.input_tokens,
        "input_tokens_actual": prompt_plan["input_tokens_actual"],
        "server_prompt_tokens_exact": server_prompt_tokens_exact,
        "tokenizer": args.tokenizer,
        "chat_template_sha256": prompt_plan["chat_template_sha256"],
        "unique_prompt_count": prompt_plan["unique_prompt_count"],
        "output_tokens_requested": args.output_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "completed": len(good),
        "failed": len(records) - len(good),
        "finish_reasons": {reason: sum(row["finish_reason"] == reason for row in good)
                           for reason in sorted({row["finish_reason"] for row in good
                                                if row["finish_reason"] is not None})},
        "ttft_ms": metric_summary(ttft),
        "tpot_ms": metric_summary(tpot),
        "e2e_ms": metric_summary(e2e),
        "output_tokens_total": sum(row["output_tokens"] for row in good),
        "wall_time_s": wall_time,
        "prefix_cache": {
            "target_pct": args.prefix_cache_pct,
            "rendered_common_prefix_tokens": prompt_plan["rendered_common_prefix_tokens"],
            "rendered_common_prefix_pct": prompt_plan["rendered_common_prefix_pct"],
            "prewarmed": prewarm_record is not None,
            "usage_details_complete": cache_details_complete,
            "cached_tokens_total": cached_tokens_total,
            "prompt_tokens_total": prompt_tokens_total,
            "actual_cached_token_ratio": actual_cache_ratio,
            "tolerance": tolerance,
            "ratio_gate": cache_ratio_gate,
        },
        "arrival": {
            "mode": args.arrival_mode,
            "stream_seed": args.stream_seed,
            "request_rate_target_rps": args.request_rate,
            "request_rate_realized_rps": ((len(starts) - 1) / (starts[-1] - starts[0])
                                          if len(starts) > 1 and starts[-1] > starts[0]
                                          else None),
            "realized_interarrival_s": metric_summary(realized_interarrivals),
            "scheduling_lag_s": metric_summary(scheduling_lag),
            "peak_in_flight": peak_in_flight(good),
        },
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(1)
    if args.require_cache_details and not cache_details_complete:
        raise SystemExit(2)
    if args.tokenizer and not server_prompt_tokens_exact:
        raise SystemExit(4)
    if args.require_cache_details and not cache_ratio_gate:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
