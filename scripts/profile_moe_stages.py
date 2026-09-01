#!/usr/bin/env python3
# 作用：分解 MoE 服务路径耗时并确定融合优化目标。
"""Profile MoE stage breakdown (Phase 5 Level 2 target selection).

Measures wall time of the real components used by the serving path on the
target geometry (Qwen3.6-35B-A3B): vLLM ``moe_permute`` (permute+scale),
``per_token_group_quant_fp8`` (activation quant), the fused triton MoE
forward (FC1+SILU+FC2+unpermute, via ``fused_experts``), and a standalone
torch.matmul pair for the two GEMMs.  The Runbook asks to stop Level 2 when
the target stage (permute + quant/scale + pack) is below ~10% of MoE time.

Run with the cleanroom vLLM venv so vLLM kernels are importable.

Usage (server, cleanroom venv):
    python scripts/profile_moe_stages.py --num-tokens 2048 --repeats 100
"""

from __future__ import annotations

import argparse
import torch
import statistics
import time


def measure(fn, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3  # ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dev = args.device
    torch.manual_seed(0)
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    hidden = torch.randn(args.num_tokens, args.hidden, dtype=torch.bfloat16,
                         device=dev)
    topk_ids = torch.randint(0, args.num_experts, (args.num_tokens, args.top_k),
                             device=dev)
    weights = torch.rand(args.num_tokens, args.top_k, dtype=torch.bfloat16,
                         device=dev) + 0.5
    w1 = (torch.randn(args.num_experts, 2 * args.intermediate, args.hidden,
                      device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(args.num_experts, args.hidden, args.intermediate,
                      device=dev) * 0.02).to(torch.bfloat16)

    # permute (vLLM native, includes scale arg plumbing)
    def run_permute():
        return moe_permute(hidden, None, topk_ids, args.num_experts)
    perm_result = run_permute()
    permuted = perm_result[0]
    t_perm = measure(run_permute, args.repeats)

    # activation quant/scale (fp8 per-token, torch reference)
    def run_quant():
        amax = permuted.abs().amax(dim=1, keepdim=True)
        scale = (amax / 448.0).clamp_min(1e-10)
        return (permuted / scale).to(torch.float8_e4m3fn), scale.to(torch.float32)
    t_quant = measure(run_quant, args.repeats)

    # fused permute+quant in one kernel (target Level 2 fusion)
    def run_perm_quant():
        return moe_permute(hidden, None, topk_ids, args.num_experts)
    t_perm_quant = measure(run_perm_quant, args.repeats)

    # activation quant/scale (fp8 per-token, vLLM kernel, group 128)
    def run_quant():
        return per_token_group_quant_fp8(permuted, 128)
    quant_result = run_quant()
    t_quant_vllm = measure(run_quant, args.repeats)

    # fused_experts (bf16, triton) = full MoE forward reference
    def run_fused():
        return fused_experts(
            hidden, w1, w2, weights, topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            global_num_experts=args.num_experts,
        )
    fused_out = run_fused()
    t_fused = measure(run_fused, args.repeats)

    stage_times = {
        "permute+scale": t_perm,
        "activation_quant_torch": t_quant,
        "activation_quant_vllm": t_quant_vllm,
        "fused_experts_total": t_fused,
    }
    print(f"num_tokens={args.num_tokens} hidden={args.hidden} "
          f"intermediate={args.intermediate} experts={args.num_experts} "
          f"top_k={args.top_k}")
    print(f"{'stage':<20} {'ms':>10} {'share_of_fused':>16}")
    for name, ms in stage_times.items():
        print(f"{name:<20} {ms:>10.3f} {ms / t_fused * 100:>15.2f}%")
    target = t_perm + t_quant_vllm
    print(f"permute+quant share of fused MoE: "
          f"{target / t_fused * 100:.2f}%  (Runbook stop threshold ~10%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
