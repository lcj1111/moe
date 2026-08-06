#!/usr/bin/env python3
"""Benchmark MoE kernels (vLLM triton fused_experts / FlashInfer cutlass).

``m_bucket`` is the total token count (the flat ``num_tokens`` input to the
MoE kernel), matching Phase 4/8 semantics where ``prefill_m = input_tokens *
concurrency`` and ``decode_m = concurrency``, both rounded to power-of-two
buckets.  The kernel receives ``num_tokens = m_bucket`` rows of hidden states;
per-expert rows are an internal detail of the routed kernel.

Latencies are real GPU measurements (kernel execution), not synthetic
estimates.  Output rows follow ``qtopomoe.kernel_measurement.v1`` and can be
merged into ``configs/kernels/phase4_kernel_db.json``.

Model geometry (Qwen3.6-35B-A3B text config):
    hidden_size=2048, moe_intermediate_size=512, num_experts=256,
    num_experts_per_tok=8, num_hidden_layers=40.

Usage (server, cleanroom venv):
    python scripts/bench_moe_kernel.py --output /tmp/moe_kernel_triton.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def make_inputs(
    num_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    device: str,
):
    torch.manual_seed(42)
    hidden_states = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    # w1: [num_experts, 2*intermediate, hidden] (gate + up fused)
    w1 = torch.randn(num_experts, 2 * intermediate, hidden, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(num_experts, hidden, intermediate, dtype=dtype, device=device) * 0.02
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    topk_weights = torch.rand(num_tokens, top_k, device=device) + 0.5
    return hidden_states, w1, w2, topk_weights, topk_ids


def measure(fn, repeats: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)  # microseconds
    ordered = sorted(times)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"p50_us": round(p50, 2), "p95_us": round(p95, 2),
            "min_us": round(ordered[0], 2), "max_us": round(ordered[-1], 2),
            "repeats": repeats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buckets", default="1,8,16,32,256,2048,8192,16384")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--backend", default="triton", choices=["triton", "cutlass"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Import inside main so --help works without the vLLM venv.
    if args.backend == "triton":
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        kernel_name = "vllm.fused_experts(triton)"
        num_experts = args.num_experts
        if args.precision == "fp8":
            from vllm.model_executor.layers.fused_moe.config import (
                fp8_w8a8_moe_quant_config,
            )
            quant_config = None
        def run(*kernel_args):
            if args.precision == "fp8":
                hidden_states, w1, w2, topk_weights, topk_ids, qc = kernel_args
            else:
                hidden_states, w1, w2, topk_weights, topk_ids = kernel_args
                qc = None
            return fused_experts(
                hidden_states, w1, w2, topk_weights, topk_ids,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
                global_num_experts=num_experts,
                quant_config=qc,
            )
    elif args.backend == "cutlass":
        from flashinfer import cutlass_fused_moe
        kernel_name = "flashinfer.cutlass_fused_moe"
        def run(*args):
            x, ids, scales, fc1, fc2 = args
            return cutlass_fused_moe(
                x, ids, scales, fc1, fc2, torch.bfloat16, [],
                tp_size=1, ep_size=1,
            )[0]
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend {args.backend}")

    device = args.device
    dtype = torch.bfloat16
    rows = []
    for m_bucket in [int(x) for x in args.buckets.split(",")]:
        base_row = {
            "schema_version": "qtopomoe.kernel_measurement.v1",
            "backend": args.backend,
            "kernel_name": kernel_name,
            "m_bucket": m_bucket,
            "precision": args.precision,
            "kernel_config": {
                "hidden": args.hidden,
                "moe_intermediate": args.intermediate,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "per_expert_tokens": m_bucket,
                "activation": "silu",
                "apply_router_weight_on_input": False,
            },
        }
        try:
            if args.backend == "triton":
                num_tokens = m_bucket
                if args.precision == "fp8":
                    torch.manual_seed(42)
                    hidden_states = torch.randn(
                        num_tokens, args.hidden, dtype=dtype, device=device)
                    w1 = (torch.randn(args.num_experts, 2 * args.intermediate,
                                      args.hidden, device=device) * 0.02).to(
                        torch.float8_e4m3fn)
                    w2 = (torch.randn(args.num_experts, args.hidden,
                                      args.intermediate, device=device) * 0.02).to(
                        torch.float8_e4m3fn)
                    s1 = torch.ones(args.num_experts, 1, 1, dtype=torch.float32,
                                    device=device)
                    s2 = torch.ones(args.num_experts, 1, 1, dtype=torch.float32,
                                    device=device)
                    a1 = torch.tensor(1.0, dtype=torch.float32, device=device)
                    qc = fp8_w8a8_moe_quant_config(
                        w1_scale=s1, w2_scale=s2, a1_scale=a1, a2_scale=a1)
                    topk_ids = torch.randint(
                        0, args.num_experts, (num_tokens, args.top_k), device=device)
                    topk_weights = torch.ones(
                        num_tokens, args.top_k, dtype=dtype, device=device)
                    inputs = (hidden_states, w1, w2, topk_weights, topk_ids, qc)
                else:
                    hidden_states, w1, w2, topk_weights, topk_ids = make_inputs(
                        num_tokens, args.hidden, args.intermediate, args.num_experts,
                        args.top_k, dtype, device,
                    )
                    inputs = (hidden_states, w1, w2, topk_weights, topk_ids)
                out = run(*inputs)
                expected = (num_tokens, args.hidden)
            else:  # cutlass
                num_tokens = m_bucket
                x = torch.randn(num_tokens, args.hidden, dtype=dtype, device=device)
                ids = torch.randint(0, args.num_experts, (num_tokens, args.top_k),
                                    dtype=torch.int32, device=device)
                scales = torch.ones(num_tokens, args.top_k, dtype=torch.float32,
                                    device=device)
                fc1 = torch.randn(args.num_experts, 2 * args.intermediate, args.hidden,
                                  dtype=dtype, device=device) * 0.02
                fc2 = torch.randn(args.num_experts, args.hidden, args.intermediate,
                                  dtype=dtype, device=device) * 0.02
                inputs = (x, ids, scales, fc1, fc2)
                out = run(*inputs)
                expected = (num_tokens, args.hidden)
            torch.cuda.synchronize()
            if tuple(out.shape) != expected:
                raise RuntimeError(f"unexpected output shape {tuple(out.shape)} != {expected}")
            timing = measure(lambda: run(*inputs), args.repeats)
            row = {**base_row, "measured": True, "valid": True,
                   "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
                   "source": "phase4_bench_20260806", **timing}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        except Exception as exc:  # record a failed row without aborting the sweep
            row = {**base_row, "measured": False, "valid": False,
                   "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
                   "source": "phase4_bench_20260806",
                   "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        finally:
            if "inputs" in locals():
                del inputs
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
