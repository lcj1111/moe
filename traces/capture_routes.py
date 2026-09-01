#!/usr/bin/env python3
# 作用：通过 vLLM 原生接口采集逐 token、逐层专家路由 ID。
"""Capture per-token MoE expert routing traces using vLLM's native
``enable_return_routed_experts`` mechanism.

Captured data: per ``(token_position, layer)`` top-k expert IDs
(uint8/uint16 array of shape ``[seq_len, num_layers, top_k]``).

NOT captured by this mechanism (recorded as ``null`` in the trace rows):
  - ``topk_weights`` (router probabilities are not exported);
  - ``expert_service_time_us`` (requires kernel-level profiling, Phase 4).

The script runs its own vLLM engine (cleanroom runtime, upstream native
flag, no private adapter). It does not touch running services.

Outputs under ``--output-dir``:
  traces/<prompt_id>.npy      per-prompt ndarray [seq_len, layers, topk]
  traces.jsonl                runbook schema rows (one per token/layer)
  expert_token_histogram.json per-layer per-expert token counts + M buckets
  capture_manifest.json       frozen params, token-ID hashes, file hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    """Render a manifest row to the exact string sent to the model.

    Supports either an explicit raw ``prompt`` or chat ``messages``
    (rendered with the checkpoint's chat template, mirroring the frozen
    official-like protocol incl. ``enable_thinking``).
    """
    if "prompt" in row:
        return str(row["prompt"])
    messages = row["messages"]
    sampling = row.get("sampling", {})
    kwargs = {"enable_thinking": bool(sampling.get("enable_thinking", False))}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=kwargs,
    )


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def m_bucket(value: int) -> str:
    bounds = [1, 5, 9, 17, 33, 65, 129, 257]
    labels = ["0", "1-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", ">256"]
    for i, upper in enumerate(bounds):
        if value < upper:
            return labels[i]
    return labels[-1]


def build_histogram(
    arrays: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    """Aggregate per-layer per-expert token counts and M-bucket weights."""
    per_layer: list[dict[str, Any]] = []
    total_tokens = 0
    if not arrays:
        return {"per_layer": [], "total_tokens": 0, "m_buckets": {}}
    _, sample = arrays[0]
    num_layers = int(sample.shape[1])
    top_k = int(sample.shape[2])
    for layer in range(num_layers):
        counts: dict[int, int] = defaultdict(int)
        for _, arr in arrays:
            for expert in arr[:, layer, :].reshape(-1).tolist():
                counts[int(expert)] += 1
        bucket_counts: dict[str, int] = defaultdict(int)
        for expert, count in counts.items():
            bucket_counts[m_bucket(count)] += 1
        per_layer.append(
            {
                "layer_id": layer,
                "expert_counts": {str(k): v for k, v in sorted(counts.items())},
                "m_bucket_experts": dict(sorted(bucket_counts.items())),
                "sum_tokens": sum(counts.values()),
            }
        )
        total_tokens += sum(counts.values())
    global_buckets: dict[str, int] = defaultdict(int)
    for layer_info in per_layer:
        for bucket, count in layer_info["m_bucket_experts"].items():
            global_buckets[bucket] += count
    return {
        "per_layer": per_layer,
        "total_tokens": total_tokens,
        "num_layers": num_layers,
        "top_k": top_k,
        "m_buckets": dict(sorted(global_buckets.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quant-format", required=True)
    parser.add_argument("--prompt-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--indices", required=True, help="comma-separated 0-based row indices")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--routed-experts-prompt-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--max-prompt-tokens", type=int, default=0, help="0 = no truncation")
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    rows_all = load_rows(args.prompt_manifest)
    rows = [rows_all[i] for i in indices]

    # Delayed imports so --help works without the cleanroom runtime.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    rendered = [render_prompt(row, tokenizer) for row in rows]

    prompt_info: list[dict[str, Any]] = []
    prompts: list[str] = []
    for row, text in zip(rows, rendered):
        ids = tokenizer.encode(text)
        if args.max_prompt_tokens and len(ids) > args.max_prompt_tokens:
            ids = ids[: args.max_prompt_tokens]
        prompt_tokens = [int(x) for x in ids]
        prompt_info.append(
            {
                "prompt_id": str(row["id"]),
                "prompt_tokens": len(prompt_tokens),
                "prompt_token_ids_sha256": sha256_bytes(
                    json.dumps(prompt_tokens, separators=(",", ":")).encode("ascii")
                ),
            }
        )
        prompts.append(tokenizer.decode(prompt_tokens))

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_return_routed_experts=True,
        moe_backend=args.moe_backend,
    )
    params = SamplingParams(
        temperature=0,
        seed=args.seed,
        max_tokens=args.gen_tokens,
        routed_experts_prompt_start=args.routed_experts_prompt_start,
    )
    outputs = llm.generate(prompts, params, use_tqdm=True)

    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    arrays: list[tuple[str, np.ndarray]] = []
    file_records: list[dict[str, Any]] = []

    for row, info, output in zip(rows, prompt_info, outputs):
        re = output.outputs[0].routed_experts
        if re is None:
            print(f"WARNING routed_experts is None for {info['prompt_id']}", file=sys.stderr)
            continue
        prompt_id = info["prompt_id"]
        arr = np.asarray(re)
        npy_path = trace_dir / f"{prompt_id}.npy"
        np.save(npy_path, arr)
        file_records.append(
            {
                "prompt_id": prompt_id,
                "file": str(npy_path.relative_to(args.output_dir)),
                "bytes": npy_path.stat().st_size,
                "sha256": sha256_bytes(npy_path.read_bytes()),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        )
        arrays.append((prompt_id, arr))
        num_layers = int(arr.shape[1])
        for token_pos in range(int(arr.shape[0])):
            for layer in range(num_layers):
                ids = [int(x) for x in arr[token_pos, layer, :].tolist()]
                trace_rows.append(
                    {
                        "trace_id": f"{args.model_id}:{prompt_id}",
                        "model_id": args.model_id,
                        "quant_format": args.quant_format,
                        "prompt_id": prompt_id,
                        "token_position": token_pos,
                        "layer_id": layer,
                        "topk_expert_ids": ids,
                        "topk_weights": None,
                        "expert_service_time_us": None,
                    }
                )

    trace_path = args.output_dir / "traces.jsonl"
    write_jsonl(trace_path, trace_rows)
    histogram = build_histogram(arrays)
    histogram_path = args.output_dir / "expert_token_histogram.json"
    histogram_path.write_text(json.dumps(histogram, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "qtopomoe.route_trace_capture.v1",
        "model_id": args.model_id,
        "quant_format": args.quant_format,
        "model_path": args.model_path,
        "prompt_manifest": str(args.prompt_manifest),
        "indices": indices,
        "seed": args.seed,
        "gen_tokens": args.gen_tokens,
        "routed_experts_prompt_start": args.routed_experts_prompt_start,
        "max_prompt_tokens": args.max_prompt_tokens,
        "gpu_ids": args.gpu_ids,
        "tp_size": args.tp_size,
        "max_model_len": args.max_model_len,
        "moe_backend": args.moe_backend,
        "prompts": prompt_info,
        "traces": file_records,
        "traces_rows": len(trace_rows),
        "traces_jsonl_sha256": sha256_bytes(trace_path.read_bytes()),
        "histogram_sha256": sha256_bytes(histogram_path.read_bytes()),
        "note": (
            "topk_weights and expert_service_time_us are null: not exported by "
            "vLLM enable_return_routed_experts."
        ),
    }
    manifest_path = args.output_dir / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "prompts": len(arrays),
                "trace_rows": len(trace_rows),
                "histogram": histogram,
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
