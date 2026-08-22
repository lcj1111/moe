#!/usr/bin/env python3
"""Benchmark MoE kernels (vLLM Triton/WNA16 / FlashInfer CUTLASS).

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
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--backend",
        default="triton",
        choices=["triton", "wna16_triton", "cutlass", "nvfp4_cutlass"],
    )
    parser.add_argument(
        "--precision", default="bf16", choices=["bf16", "fp8", "w4a16", "nvfp4"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source", default="phase4_bench_20260809")
    args = parser.parse_args()

    if (args.backend == "nvfp4_cutlass") != (args.precision == "nvfp4"):
        raise SystemExit("backend=nvfp4_cutlass and precision=nvfp4 must be used together")
    if (args.backend == "wna16_triton") != (args.precision == "w4a16"):
        raise SystemExit(
            "backend=wna16_triton and precision=w4a16 must be used together"
        )
    if args.precision == "w4a16" and (
        args.hidden % args.group_size or args.intermediate % args.group_size
    ):
        raise SystemExit(
            "W4A16 hidden and intermediate sizes must be divisible by group-size"
        )

    # Import inside main so --help works without the vLLM venv.
    w4_contract_validator = None
    if args.backend in ("triton", "wna16_triton"):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        kernel_name = (
            "vllm.fused_experts(wna16_triton)"
            if args.backend == "wna16_triton"
            else "vllm.fused_experts(triton)"
        )
        num_experts = args.num_experts
        if args.precision == "fp8":
            from vllm.model_executor.layers.fused_moe.config import (
                fp8_w8a8_moe_quant_config,
            )
        elif args.precision == "w4a16":
            from vllm.model_executor.layers.fused_moe.config import (
                int4_w4a16_moe_quant_config,
            )

        def run(*kernel_args):
            if args.precision in ("fp8", "w4a16"):
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

        if args.precision == "w4a16":
            def validate_w4a16_contract(device: str) -> dict[str, float | bool]:
                """Compare packed WNA16 output with explicit dequantization.

                This small, deterministic reference Gate verifies nibble order,
                symmetric zero-point convention, group scale application, and
                SiLU-and-multiply semantics before recording performance rows.
                """
                torch.manual_seed(17)
                m, hidden, intermediate, experts = 3, 128, 128, 2
                x = torch.randn(m, hidden, dtype=torch.bfloat16, device=device)
                w1 = torch.randint(
                    0, 256, (experts, 2 * intermediate, hidden // 2),
                    dtype=torch.uint8, device=device,
                )
                w2 = torch.randint(
                    0, 256, (experts, hidden, intermediate // 2),
                    dtype=torch.uint8, device=device,
                )
                s1 = torch.full(
                    (experts, 2 * intermediate, 1), 0.002,
                    dtype=torch.bfloat16, device=device,
                )
                s2 = torch.full(
                    (experts, hidden, 1), 0.002,
                    dtype=torch.bfloat16, device=device,
                )
                ids = torch.zeros((m, 1), dtype=torch.int64, device=device)
                weights = torch.ones((m, 1), dtype=torch.float32, device=device)
                qc = int4_w4a16_moe_quant_config(
                    w1_scale=s1, w2_scale=s2, block_shape=[0, 128]
                )
                actual = fused_experts(
                    x, w1, w2, weights, ids,
                    activation=MoEActivation.SILU,
                    apply_router_weight_on_input=False,
                    global_num_experts=experts,
                    quant_config=qc,
                )

                def unpack_centered_int4(packed: torch.Tensor) -> torch.Tensor:
                    low = (packed.to(torch.int16) & 0xF) - 8
                    high = ((packed.to(torch.int16) >> 4) & 0xF) - 8
                    return torch.stack((low, high), dim=-1).reshape(
                        *packed.shape[:-1], packed.shape[-1] * 2
                    )

                dense_w1 = unpack_centered_int4(w1[0]).float() * s1[0].float()
                dense_w2 = unpack_centered_int4(w2[0]).float() * s2[0].float()
                gate_up = x.float() @ dense_w1.T
                reference = (
                    torch.nn.functional.silu(gate_up[:, :intermediate])
                    * gate_up[:, intermediate:]
                ) @ dense_w2.T
                torch.cuda.synchronize()
                delta = (actual.float() - reference).abs()
                max_abs_error = float(delta.max().item())
                mean_abs_error = float(delta.mean().item())
                accepted = bool(torch.allclose(
                    actual.float(), reference, atol=5e-4, rtol=5e-2
                ))
                if not accepted:
                    raise RuntimeError(
                        "W4A16 packed-contract reference check failed: "
                        f"max_abs_error={max_abs_error}"
                    )
                return {
                    "accepted": accepted,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                    "reference": "explicit_centered_int4_dequant_bf16",
                }

            w4_contract_validator = validate_w4a16_contract
    elif args.backend == "cutlass":
        from flashinfer import cutlass_fused_moe
        kernel_name = "flashinfer.cutlass_fused_moe"
        def run(*args):
            x, ids, scales, fc1, fc2 = args
            return cutlass_fused_moe(
                x, ids, scales, fc1, fc2, torch.bfloat16, [],
                tp_size=1, ep_size=1,
            )[0]
    elif args.backend == "nvfp4_cutlass":
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            run_cutlass_moe_fp4,
        )
        kernel_name = "vllm.run_cutlass_moe_fp4"

        def run(*kernel_args):
            (
                output, hidden_states, a1_gscale, w1, w1_scale, w1_alphas,
                a2_gscale, w2, w2_scale, w2_alphas, topk_weights, topk_ids,
                workspace13, workspace2,
            ) = kernel_args
            return run_cutlass_moe_fp4(
                output=output,
                a=hidden_states,
                a1_gscale=a1_gscale,
                w1_fp4=w1,
                w1_blockscale=w1_scale,
                w1_alphas=w1_alphas,
                a2_gscale=a2_gscale,
                w2_fp4=w2,
                w2_blockscale=w2_scale,
                w2_alphas=w2_alphas,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                workspace13=workspace13,
                workspace2=workspace2,
                m=hidden_states.shape[0],
                n=args.intermediate,
                k=args.hidden,
                e=args.num_experts,
                device=hidden_states.device,
                apply_router_weight_on_input=False,
            )
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend {args.backend}")

    device = args.device
    dtype = torch.bfloat16
    w4_contract_evidence = (
        w4_contract_validator(device) if w4_contract_validator is not None else None
    )
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
                "total_tokens": m_bucket,
                "activation": "silu",
                "apply_router_weight_on_input": False,
            },
        }
        if args.precision == "w4a16":
            base_row["kernel_config"].update({
                "group_size": args.group_size,
                "weight_dtype": "uint8_packed_centered_int4",
                "weight_layout": "N_first_two_nibbles_per_byte",
                "checkpoint_source_layout": "int32_K_first_pack_quantized",
            })
            base_row["packed_contract_validation"] = w4_contract_evidence
        try:
            if args.backend in ("triton", "wna16_triton"):
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
                elif args.precision == "w4a16":
                    # This is the actual vLLM Triton WNA16 contract used by the
                    # frozen Q-TopoMoE service.  Compressed-tensors checkpoints
                    # store int32 K-first GPTQ packing; vLLM transposes it and
                    # views the bytes as uint8 N-first, with two centered int4
                    # nibbles per byte.  Scales are groupwise BF16 tensors.
                    torch.manual_seed(42)
                    hidden_states = torch.randn(
                        num_tokens, args.hidden, dtype=dtype, device=device
                    )
                    w1 = torch.randint(
                        0,
                        256,
                        (
                            args.num_experts,
                            2 * args.intermediate,
                            args.hidden // 2,
                        ),
                        dtype=torch.uint8,
                        device=device,
                    )
                    w2 = torch.randint(
                        0,
                        256,
                        (
                            args.num_experts,
                            args.hidden,
                            args.intermediate // 2,
                        ),
                        dtype=torch.uint8,
                        device=device,
                    )
                    # 0.002 keeps random packed weights in the same useful
                    # numerical range as the model while correctness is gated
                    # on shape and finite output rather than model quality.
                    s1 = torch.full(
                        (
                            args.num_experts,
                            2 * args.intermediate,
                            args.hidden // args.group_size,
                        ),
                        0.002,
                        dtype=dtype,
                        device=device,
                    )
                    s2 = torch.full(
                        (
                            args.num_experts,
                            args.hidden,
                            args.intermediate // args.group_size,
                        ),
                        0.002,
                        dtype=dtype,
                        device=device,
                    )
                    qc = int4_w4a16_moe_quant_config(
                        w1_scale=s1,
                        w2_scale=s2,
                        block_shape=[0, args.group_size],
                    )
                    topk_ids = torch.randint(
                        0,
                        args.num_experts,
                        (num_tokens, args.top_k),
                        dtype=torch.int64,
                        device=device,
                    )
                    topk_weights = torch.full(
                        (num_tokens, args.top_k),
                        1.0 / args.top_k,
                        dtype=torch.float32,
                        device=device,
                    )
                    inputs = (hidden_states, w1, w2, topk_weights, topk_ids, qc)
                else:
                    hidden_states, w1, w2, topk_weights, topk_ids = make_inputs(
                        num_tokens, args.hidden, args.intermediate, args.num_experts,
                        args.top_k, dtype, device,
                    )
                    inputs = (hidden_states, w1, w2, topk_weights, topk_ids)
                out = run(*inputs)
                expected = (num_tokens, args.hidden)
            elif args.backend == "cutlass":
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
            else:  # native VLLM_CUTLASS NVFP4 MoE
                num_tokens = m_bucket
                hidden_states = torch.randn(
                    num_tokens, args.hidden, dtype=dtype, device=device)
                topk_ids = torch.randint(
                    0, args.num_experts, (num_tokens, args.top_k),
                    dtype=torch.int32, device=device)
                topk_weights = torch.full(
                    (num_tokens, args.top_k), 1.0 / args.top_k,
                    dtype=dtype, device=device)
                w1 = torch.randint(
                    0, 256,
                    (args.num_experts, 2 * args.intermediate, args.hidden // 2),
                    dtype=torch.uint8, device=device)
                w2 = torch.randint(
                    0, 256,
                    (args.num_experts, args.hidden, args.intermediate // 2),
                    dtype=torch.uint8, device=device)
                w1_scale = torch.full(
                    (args.num_experts, 2 * args.intermediate, args.hidden // 16),
                    0.015625, dtype=torch.float8_e4m3fn, device=device)
                w2_scale = torch.full(
                    (args.num_experts, args.hidden, args.intermediate // 16),
                    0.015625, dtype=torch.float8_e4m3fn, device=device)
                w1_alphas = torch.ones(args.num_experts, dtype=torch.float32, device=device)
                w2_alphas = torch.ones(args.num_experts, dtype=torch.float32, device=device)
                a1_gscale = torch.ones(args.num_experts, dtype=torch.float32, device=device)
                a2_gscale = torch.ones(args.num_experts, dtype=torch.float32, device=device)
                workspace13 = torch.empty(
                    num_tokens * args.top_k,
                    max(2 * args.intermediate, args.hidden),
                    dtype=dtype, device=device)
                workspace2 = torch.empty(
                    num_tokens * args.top_k, args.intermediate,
                    dtype=dtype, device=device)
                output = torch.empty(num_tokens, args.hidden, dtype=dtype, device=device)
                inputs = (
                    output, hidden_states, a1_gscale, w1, w1_scale, w1_alphas,
                    a2_gscale, w2, w2_scale, w2_alphas, topk_weights, topk_ids,
                    workspace13, workspace2,
                )
                run(*inputs)
                out = output
                expected = (num_tokens, args.hidden)
            torch.cuda.synchronize()
            if tuple(out.shape) != expected:
                raise RuntimeError(f"unexpected output shape {tuple(out.shape)} != {expected}")
            if not torch.isfinite(out).all():
                raise RuntimeError("kernel output contains non-finite values")
            output_abs_max = float(out.abs().max().item())
            timing = measure(lambda: run(*inputs), args.repeats)
            row = {**base_row, "measured": True, "valid": True,
                   "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
                   "source": args.source,
                   "correctness": {
                       "output_shape": list(expected),
                       "output_finite": True,
                       "output_abs_max": output_abs_max,
                   },
                   **timing}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        except Exception as exc:  # record a failed row without aborting the sweep
            row = {**base_row, "measured": False, "valid": False,
                   "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
                   "source": args.source,
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
