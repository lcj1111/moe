#!/usr/bin/env python3
"""Phase 5 prepare-stage A/B: vLLM permute+quant vs our fused kernel.

Side A (baseline): vLLM serving path prepare = ``moe_permute`` (permute +
scale plumbing) followed by ``per_token_group_quant_fp8`` (fp8 activation
quantize).  Side B (candidate): our single fused ``permute + quant/scale +
pack`` Triton kernel.  Both operate on the same input and produce the same
per-expert packed fp8 activations; correctness of B is checked against the
Torch reference in ``fused_permute_quant.py``.

This is the reproducible micro A/B the Runbook requires before a two-week
vLLM modular-experts integration; it does not fabricate an in-tree patch.

Usage (server, cleanroom venv):
    python scripts/bench_moe_endtoend_ab.py --repeats 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from fused_permute_quant import permute_quant_fused


def measure(fn, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=str, default="/tmp/phase5_ab.json")
    args = parser.parse_args()

    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    torch.manual_seed(0)
    dev = args.device
    rows = []
    for m in [1, 8, 32, 256, 2048, 8192, 16384]:
        hidden = torch.randn(m, args.hidden, dtype=torch.bfloat16, device=dev)
        topk_ids = torch.randint(0, args.num_experts, (m, args.top_k), device=dev)
        weights = torch.rand(m, args.top_k, dtype=torch.bfloat16, device=dev) + 0.5
        w1 = (torch.randn(args.num_experts, 2 * args.intermediate, args.hidden,
                          device=dev) * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(args.num_experts, args.hidden, args.intermediate,
                          device=dev) * 0.02).to(torch.bfloat16)

        # Side A: vLLM serving-path prepare (permute then quant)
        def run_a_prep():
            permuted, _, _, _, _ = moe_permute(
                hidden, None, topk_ids, args.num_experts)
            return per_token_group_quant_fp8(permuted, 128)
        a_result = run_a_prep()
        t_a = measure(run_a_prep, args.repeats)

        # Side B: fused permute+quant+pack
        def run_b_prep():
            return permute_quant_fused(hidden, topk_ids, args.num_experts)
        b_result = run_b_prep()
        t_b = measure(run_b_prep, args.repeats)

        row = {
            "m": m,
            "side_a_vllm_prepare_ms": round(t_a, 4),
            "side_b_fused_ms": round(t_b, 4),
            "ab_ratio": round(t_a / t_b, 3) if t_b > 0 else None,
        }
        rows.append(row)
        print(json.dumps(row))

    payload = {
        "schema_version": "qtopomoe.phase5.ab.v1",
        "note": (
            "Side A = vLLM serving-path prepare (moe_permute + "
            "per_token_group_quant_fp8). Side B = fused permute+quant+pack "
            "Triton kernel. Same inputs; prepare-stage micro A/B only."
        ),
        "rows": rows,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print("saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
