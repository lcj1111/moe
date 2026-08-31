#!/usr/bin/env python3
# 作用：实现并验证融合 permute、激活量化和打包的 Triton kernel。
"""Phase 5 Level 2: fused permute + activation quant/scale + pack (Triton).

Target stage from profiling: permute (8.7%) + activation quant (4.6%) =
13.3% of the fused triton MoE forward, above the Runbook 10% stop threshold.
This module implements a single Triton kernel that, given bf16 hidden states
and top-k expert ids, writes per-expert packed fp8(e4m3) activations plus
per-token scales in one pass, and benchmarks it against the separable Torch
reference (permute then quant).

Correctness is checked against the Torch reference (dequantized fp8 vs bf16,
within fp8 rounding tolerance) across several shapes including extreme
decode-like (M=1) and prefill-like (M=16384) batches.

Usage (server, cleanroom venv):
    python scripts/fused_permute_quant.py --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_permute_quant_kernel(
    hidden_ptr,  # (M, K) bf16
    topk_ids_ptr,  # (M, topk) int32
    expert_offset_ptr,  # (num_experts+1,) int32 exclusive prefix sum
    out_q_ptr,  # (M*topk, K) fp8
    out_s_ptr,  # (M*topk,) fp32 per-token scale
    out_pos_ptr,  # (M*topk,) int32 position per (token, expert) slot
    M,
    K,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row < M
    # Pass 1: per-token abs max over the full row.
    row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = kk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = (k_offs < K)[None, :] & row_mask[:, None]
        x = tl.load(hidden_ptr + row[:, None] * K + k_offs[None, :],
                    mask=k_mask, other=0.0).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(tl.abs(x), axis=1))
    scale = row_max / 448.0
    scale = tl.where(scale < 1e-10, 1.0, scale)
    # Pass 2: quantize and store to the per-expert packed slots.
    for j in tl.static_range(topk):
        e = tl.load(topk_ids_ptr + row * topk + j, mask=row_mask, other=0)
        off = tl.load(expert_offset_ptr + e, mask=row_mask, other=0)
        pos = tl.load(out_pos_ptr + row * topk + j, mask=row_mask, other=0)
        slot = off + pos
        tl.store(out_s_ptr + slot, scale, mask=row_mask)
        for kk in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = kk * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = (k_offs < K)[None, :] & row_mask[:, None]
            x = tl.load(hidden_ptr + row[:, None] * K + k_offs[None, :],
                        mask=k_mask, other=0.0)
            q = (x / scale[:, None]).to(tl.float8e4nv)
            tl.store(out_q_ptr + slot[:, None] * K + k_offs[None, :], q,
                     mask=k_mask)


def permute_quant_reference(
    hidden: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference: sort expanded rows by expert, then per-token quant."""
    m, k = hidden.shape
    topk = topk_ids.shape[1]
    expanded = hidden.repeat_interleave(topk, dim=0)
    order = torch.argsort(topk_ids.reshape(-1), stable=True)
    permuted = expanded[order]
    amax = permuted.abs().amax(dim=1, keepdim=True).to(torch.float32)
    scale = (amax / 448.0).clamp_min(1e-10)
    q = (permuted.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return q, scale, order


def permute_quant_fused(
    hidden: torch.Tensor, topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused kernel: histogram -> prefix sum -> permute+quant in one pass."""
    m, k = hidden.shape
    topk = topk_ids.shape[1]
    dev = hidden.device
    flat_ids = topk_ids.reshape(-1).to(torch.int32)
    hist = torch.bincount(flat_ids, minlength=num_experts).to(torch.int32)
    expert_offset = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
    expert_offset[1:] = torch.cumsum(hist, dim=0)
    # position of each (row, j) within its expert bucket
    bucket_pos = torch.zeros(m * topk, dtype=torch.int64, device=dev)
    order = torch.argsort(flat_ids, stable=True)
    bucket_pos[order] = torch.arange(m * topk, device=dev) - expert_offset[
        flat_ids[order].long()]
    out_q = torch.empty(m * topk, k, dtype=torch.float8_e4m3fn, device=dev)
    out_s = torch.empty(m * topk, dtype=torch.float32, device=dev)
    grid = (triton.cdiv(m, 32),)
    _fused_permute_quant_kernel[grid](
        hidden, flat_ids, expert_offset, out_q, out_s, bucket_pos, m, k,
        topk, num_experts, BLOCK_M=32, BLOCK_K=128,
    )
    return out_q, out_s, order


def measure(fn, repeats: int, warmup: int = 10) -> float:
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


def check_correct(hidden, topk_ids, num_experts) -> tuple[bool, float]:
    q_ref, s_ref, _ = permute_quant_reference(hidden, topk_ids)
    q_fused, s_fused, _ = permute_quant_fused(hidden, topk_ids, num_experts)
    # dequantize both and compare
    dq_ref = q_ref.to(torch.float32) * s_ref
    dq_fused = q_fused.to(torch.float32) * s_fused.unsqueeze(-1)
    max_err = (dq_ref - dq_fused).abs().max().item()
    return max_err < 0.02, max_err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = args.device
    results = []
    for m in [1, 8, 32, 256, 2048, 8192, 16384]:
        hidden = torch.randn(m, args.hidden, dtype=torch.bfloat16, device=dev)
        topk_ids = torch.randint(0, args.num_experts, (m, args.top_k), device=dev)
        ok, err = check_correct(hidden, topk_ids, args.num_experts)
        t_ref = measure(
            lambda: permute_quant_reference(hidden, topk_ids), args.repeats)
        t_fused = measure(
            lambda: permute_quant_fused(hidden, topk_ids, args.num_experts),
            args.repeats)
        speedup = t_ref / t_fused
        results.append((m, ok, err, t_ref, t_fused, speedup))
        print(f"M={m:>6} correct={ok} max_err={err:.4f} "
              f"ref={t_ref:.4f}ms fused={t_fused:.4f}ms speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
