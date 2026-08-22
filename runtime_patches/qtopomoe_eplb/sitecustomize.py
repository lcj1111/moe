"""Opt-in bridge from a frozen Q-TopoMoE placement plan to native vLLM EPLB.

Activated only when QTOPOMOE_EPLB_PLAN points to a generated runtime plan.
The patch does not implement migration: vLLM's native synchronous EPLB owns
weight redistribution, barriers, and map commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _activate() -> None:
    plan_name = os.environ.get("QTOPOMOE_EPLB_PLAN")
    if not plan_name:
        return

    plan_path = Path(plan_name).resolve()
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    mapping = document["physical_to_logical_map"]
    actual_sha = _canonical_sha256(mapping)
    expected_sha = document.get("physical_to_logical_map_sha256")
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"placement plan checksum mismatch: {actual_sha} != {expected_sha}")

    import torch
    from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )

    frozen_map = torch.tensor(mapping, dtype=torch.int64, device="cpu")
    defer_calls = int(os.environ.get("QTOPOMOE_EPLB_DEFER_CALLS", "0"))
    invocation_count = 0

    def _supports_eplb(_self: object) -> bool:
        return True

    def _fixed_rebalance(
        cls: type,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal invocation_count
        del cls, num_groups, num_nodes
        invocation_count += 1
        expected_shape = (int(weight.shape[0]), int(num_replicas))
        if tuple(frozen_map.shape) != expected_shape:
            raise RuntimeError(
                f"placement plan shape {tuple(frozen_map.shape)} != runtime {expected_shape}"
            )
        if num_replicas % num_ranks:
            raise RuntimeError(f"{num_replicas=} is not divisible by {num_ranks=}")
        logical_experts = int(weight.shape[1])
        if invocation_count <= defer_calls:
            if old_global_expert_indices is not None:
                deferred_map = old_global_expert_indices.detach().cpu().to(torch.int64)
            else:
                deferred_map = torch.arange(num_replicas, dtype=torch.int64).repeat(
                    int(weight.shape[0]), 1
                )
            print(
                "[QTOPOMOE_EPLB_PLAN_DEFERRED] "
                f"call={invocation_count} defer_calls={defer_calls} shape={tuple(deferred_map.shape)}",
                file=sys.stderr,
                flush=True,
            )
            return deferred_map
        for layer, row in enumerate(frozen_map):
            counts = torch.bincount(row, minlength=logical_experts)
            if row.min().item() < 0 or row.max().item() >= logical_experts:
                raise RuntimeError(f"layer {layer}: logical expert id out of range")
            if not torch.all(counts == 1):
                raise RuntimeError(f"layer {layer}: expected a permutation without replicas")
        print(
            "[QTOPOMOE_EPLB_PLAN_APPLIED] "
            f"call={invocation_count} path={plan_path} sha256={actual_sha} "
            f"shape={tuple(frozen_map.shape)}",
            file=sys.stderr,
            flush=True,
        )
        return frozen_map.clone()

    CompressedTensorsW4A4Nvfp4MoEMethod.supports_eplb = property(_supports_eplb)
    DefaultEplbPolicy.rebalance_experts = classmethod(_fixed_rebalance)
    print(
        "[QTOPOMOE_EPLB_PATCH_ACTIVE] native_migration=true "
        f"plan_sha256={actual_sha}",
        file=sys.stderr,
        flush=True,
    )


_activate()
