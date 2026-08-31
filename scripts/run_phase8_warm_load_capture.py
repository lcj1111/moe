#!/usr/bin/env python3
# 作用：采集 identity placement 下的暖态逐层专家负载。
"""在 identity placement 下采集最终代表负载的真实暖态逐层专家计数。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from run_phase8_selector_limited_canary import (
    atomic_json,
    capture_service,
    load,
    restore_service,
    run_phase,
    sha256,
    start_canary,
    stop_group,
    wait_canary_health,
    wait_gpus_idle,
    wait_service_idle,
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load(args.plan)
    if plan.get("status") != "frozen_before_capture":
        raise RuntimeError("暖态采集计划未冻结")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "status.json"

    def update(stage: str, **extra: Any) -> None:
        atomic_json(status_path, {
            "schema_version": "qtopomoe.phase8_warm_load_capture_status.v1",
            "stage": stage,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            **extra,
        })

    runtime = plan["运行环境"]
    runtime_patch = repo / runtime["runtime_patch"]
    checks = {
        "python": Path(runtime["python_bin"]).exists(),
        "model": Path(runtime["model"]).exists(),
        "runtime_patch": runtime_patch.exists()
        and sha256(runtime_patch) == runtime["runtime_patch_sha256"],
    }
    if not all(checks.values()):
        update("preflight_failed", checks=checks)
        return 2

    identity = [list(range(int(runtime["num_logical_experts"]))) for _ in range(
        int(runtime["num_moe_layers"])
    )]
    identity_hash = canonical_sha256(identity)
    if identity_hash != runtime["identity_map_sha256"]:
        raise RuntimeError("identity map 哈希与冻结计划不一致")
    runtime_plan = root / "identity_runtime_plan.json"
    atomic_json(runtime_plan, {
        "schema_version": "qtopomoe.runtime_placement.v2",
        "candidate_id": "identity_warm_capture_only",
        "status": "capture_only",
        "num_layers": len(identity),
        "num_logical_experts": len(identity[0]),
        "num_ranks": runtime["tensor_parallel_size"],
        "physical_to_logical_map": identity,
        "physical_to_logical_map_sha256": identity_hash,
    })
    runtime["runtime_plan"] = str(runtime_plan)
    runtime["runtime_map_sha256"] = identity_hash
    runtime["load_export_path"] = str(root / "warm_load_windows.jsonl")
    atomic_json(root / "capture_plan.snapshot.json", plan)

    manifest = None
    process = None
    pipeline_error = None
    rollback = None
    summaries: dict[str, Any] = {}
    try:
        update("waiting_existing_service_idle", checks=checks)
        wait_service_idle(plan, update)
        manifest = capture_service(plan, root)
        update("stopping_existing_service", original_pid=manifest["original_pid"])
        stop_group(int(manifest["original_pgid"]))
        wait_gpus_idle(list(range(8)), 600)
        update("launching_identity_capture_service")
        process = start_canary(plan, repo, root)
        wait_canary_health(process, plan)
        for phase in plan["采集阶段"]:
            update("running_capture_phase", phase=phase)
            run_phase(plan, repo, root, phase)
            summaries[phase] = load(root / phase / "summary.json")
        update("capture_workload_completed")
    except BaseException as error:
        pipeline_error = repr(error)
        update("capture_failed", error=pipeline_error)
    finally:
        try:
            if process is not None:
                try:
                    pgid = os.getpgid(process.pid)
                except ProcessLookupError:
                    pgid = int(load(root / "canary_service_process.json")["pgid"])
                stop_group(pgid)
            if manifest is not None:
                update("restoring_existing_service", pipeline_error=pipeline_error)
                rollback = restore_service(manifest, plan, root)
        except BaseException as error:
            update("restore_failed", error=repr(error), pipeline_error=pipeline_error)
            return 2

    export = root / "warm_load_windows.jsonl"
    records = len(export.read_text(encoding="utf-8").splitlines()) if export.exists() else 0
    gates = {
        "pipeline_completed": pipeline_error is None,
        "all_requests_succeeded": bool(summaries) and all(
            summary.get("failed") == 0 and summary.get("completed") == summary.get("requests")
            for summary in summaries.values()
        ),
        "enough_exported_windows": records >= int(plan["gates"]["min_exported_windows"]),
        "identity_hash_consistent": identity_hash == runtime["identity_map_sha256"],
        "original_service_restored": rollback is not None
        and rollback.get("health_http") == 200
        and rollback.get("gpu_ids") == [4, 5, 6, 7],
    }
    result = {
        "schema_version": "qtopomoe.phase8_warm_load_capture_gate.v1",
        "status": "accepted" if all(gates.values()) else "rejected",
        "gates": gates,
        "pipeline_error": pipeline_error,
        "identity_map_sha256": identity_hash,
        "exported_windows": records,
        "warm_load_jsonl": str(export),
        "warm_load_jsonl_sha256": sha256(export) if export.exists() else None,
        "summaries": {
            name: {
                "requests": value.get("requests"),
                "failed": value.get("failed"),
                "p99_ms": value.get("e2e_ms", {}).get("p99"),
            }
            for name, value in summaries.items()
        },
        "rollback": rollback,
    }
    atomic_json(root / "capture_gate.json", result)
    update("completed", gate_status=result["status"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(
        RuntimeError(f"signal {signum}")
    ))
    raise SystemExit(main())
