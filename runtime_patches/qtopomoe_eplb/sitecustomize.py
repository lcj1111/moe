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
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )

    frozen_map = torch.tensor(mapping, dtype=torch.int64, device="cpu")
    defer_calls = int(os.environ.get("QTOPOMOE_EPLB_DEFER_CALLS", "0"))
    activation_name = os.environ.get("QTOPOMOE_EPLB_ACTIVATION_FILE")
    activation_path = Path(activation_name).resolve() if activation_name else None
    control_name = os.environ.get("QTOPOMOE_EPLB_CONTROL_FILE")
    control_path = Path(control_name).resolve() if control_name else None
    invocation_count = 0
    last_control_generation = 0
    original_rearrange = EplbState.rearrange

    def _control_command() -> dict[str, object] | None:
        if control_path is None or not control_path.exists():
            return None
        value = json.loads(control_path.read_text(encoding="utf-8"))
        if value.get("action") not in {"apply", "rollback"}:
            raise RuntimeError(f"unsupported placement control action: {value}")
        if int(value.get("generation", 0)) < 1:
            raise RuntimeError(f"invalid placement control generation: {value}")
        return value

    def _load_metrics(self: object) -> tuple[float, float]:
        logical_windows = []
        states = list(self.model_states.values())
        for state in states:
            physical = state.expert_load_window[:, :, : self.num_valid_physical_experts]
            logical = torch.zeros(
                self.expert_load_window_size,
                state.model.num_moe_layers,
                state.model.num_logical_experts,
                dtype=physical.dtype,
                device=physical.device,
            )
            logical.scatter_add_(
                dim=-1,
                index=state.physical_to_logical_map[
                    :, : self.num_valid_physical_experts
                ].unsqueeze(0).expand_as(physical).long(),
                src=physical,
            )
            logical_windows.append(logical.sum(dim=0))
        global_windows = self._allreduce_list(logical_windows)
        rank_loads = None
        layer_expert_cvs = []
        ep_size = get_ep_group().device_group.size()
        for state, logical in zip(states, global_windows):
            logical_float = logical.float()
            layer_means = logical_float.mean(dim=1)
            valid_layers = layer_means > 0
            if bool(valid_layers.any()):
                layer_expert_cvs.append(
                    logical_float.std(dim=1, unbiased=False)[valid_layers]
                    / layer_means[valid_layers]
                )
            mapping = state.physical_to_logical_map[:, : self.num_valid_physical_experts].long()
            counts = torch.zeros_like(logical)
            counts.scatter_add_(1, mapping, torch.ones_like(mapping, dtype=logical.dtype))
            per_physical = torch.gather(logical / counts.clamp_min(1), 1, mapping)
            values = per_physical.reshape(logical.shape[0], ep_size, -1).sum(dim=(0, 2)).float()
            rank_loads = values if rank_loads is None else rank_loads + values
        expert_cv = (
            float(torch.cat(layer_expert_cvs).median()) if layer_expert_cvs else 0.0
        )
        rank_cv = 0.0
        if rank_loads is not None and float(rank_loads.mean()) > 0:
            rank_cv = float(rank_loads.std(unbiased=False) / rank_loads.mean())
        return expert_cv, rank_cv

    def _supports_eplb(_self: object) -> bool:
        return True

    def _one_shot_rearrange(
        self: object,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        nonlocal last_control_generation
        if is_profile or rank_mapping is not None:
            return original_rearrange(self, is_profile=is_profile, rank_mapping=rank_mapping)
        if control_path is not None:
            expert_cv, rank_cv = _load_metrics(self)
            command = _control_command()
            generation = int(command["generation"]) if command else 0
            action = str(command["action"]) if command else "hold"
            print(
                "[QTOPOMOE_EPLB_LOAD_WINDOW] "
                f"call={invocation_count + 1} cv={expert_cv:.8f} "
                f"expert_cv={expert_cv:.8f} rank_cv={rank_cv:.8f} "
                f"generation={generation} action={action}",
                file=sys.stderr,
                flush=True,
            )
            if command is None or generation <= last_control_generation:
                return None
            result = original_rearrange(self, is_profile=False, rank_mapping=None)
            last_control_generation = generation
            print(
                "[QTOPOMOE_EPLB_CONTROL_COMMITTED] "
                f"generation={generation} action={action} sha256={command['target_sha256']}",
                file=sys.stderr,
                flush=True,
            )
            return result
        if activation_path is None:
            return original_rearrange(self, is_profile=False, rank_mapping=None)
        if not activation_path.exists():
            if not getattr(self, "_qtopomoe_wait_logged", False):
                print(
                    "[QTOPOMOE_EPLB_ONE_SHOT_WAITING] "
                    f"activation_file={activation_path}",
                    file=sys.stderr,
                    flush=True,
                )
                setattr(self, "_qtopomoe_wait_logged", True)
            return None
        if getattr(self, "_qtopomoe_one_shot_committed", False):
            skipped = int(getattr(self, "_qtopomoe_skipped_rearrangements", 0)) + 1
            setattr(self, "_qtopomoe_skipped_rearrangements", skipped)
            if skipped == 1:
                print(
                    "[QTOPOMOE_EPLB_ONE_SHOT_SKIPPED] reason=already_committed",
                    file=sys.stderr,
                    flush=True,
                )
            return None
        result = original_rearrange(self, is_profile=False, rank_mapping=None)
        setattr(self, "_qtopomoe_one_shot_committed", True)
        print(
            "[QTOPOMOE_EPLB_ONE_SHOT_COMMITTED] "
            f"call={invocation_count} sha256={actual_sha}",
            file=sys.stderr,
            flush=True,
        )
        return result

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
        if control_path is not None:
            command = _control_command()
            if command is None:
                return (
                    old_global_expert_indices.detach().cpu().to(torch.int64)
                    if old_global_expert_indices is not None
                    else torch.arange(num_replicas, dtype=torch.int64).repeat(
                        int(weight.shape[0]), 1
                    )
                )
            action = str(command["action"])
            generation = int(command["generation"])
            target = (
                frozen_map.clone()
                if action == "apply"
                else torch.arange(num_replicas, dtype=torch.int64).repeat(
                    int(weight.shape[0]), 1
                )
            )
            target_hash = _canonical_sha256(target.tolist())
            if command.get("target_sha256") != target_hash:
                raise RuntimeError(
                    "placement control checksum mismatch: "
                    f"{command.get('target_sha256')} != {target_hash}"
                )
            print(
                "[QTOPOMOE_EPLB_PLAN_APPLIED] "
                f"call={invocation_count} generation={generation} action={action} "
                f"sha256={target_hash} shape={tuple(target.shape)}",
                file=sys.stderr,
                flush=True,
            )
            return target
        activation_pending = activation_path is not None and not activation_path.exists()
        call_pending = activation_path is None and invocation_count <= defer_calls
        if activation_pending or call_pending:
            if old_global_expert_indices is not None:
                deferred_map = old_global_expert_indices.detach().cpu().to(torch.int64)
            else:
                deferred_map = torch.arange(num_replicas, dtype=torch.int64).repeat(
                    int(weight.shape[0]), 1
                )
            print(
                "[QTOPOMOE_EPLB_PLAN_DEFERRED] "
                f"call={invocation_count} defer_calls={defer_calls} "
                f"activation_file={activation_path} shape={tuple(deferred_map.shape)}",
                file=sys.stderr,
                flush=True,
            )
            return deferred_map
        if activation_path is not None:
            activation_hash = activation_path.read_text(encoding="utf-8").strip()
            if activation_hash != actual_sha:
                raise RuntimeError(
                    f"placement activation checksum mismatch: {activation_hash} != {actual_sha}"
                )
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
    EplbState.rearrange = _one_shot_rearrange
    print(
        "[QTOPOMOE_EPLB_PATCH_ACTIVE] native_migration=true "
        f"plan_sha256={actual_sha} activation_file={activation_path} "
        f"control_file={control_path}",
        file=sys.stderr,
        flush=True,
    )


_activate()
