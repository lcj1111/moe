#!/usr/bin/env python3
"""执行预注册的逻辑专家分布稳定性诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

from analyze_phase8_route_stability_diagnostic import analyze
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


def committed_ranks(text: str, generation: int, action: str) -> list[int]:
    return sorted(
        {
            int(rank)
            for rank in re.findall(
                rf"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_CONTROL_COMMITTED\] "
                rf"generation={generation} action={action}",
                text,
            )
        }
    )


def jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并预注册基础计划与只包含差异的确认计划。"""
    result = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_registered_plan(path: Path, repo: Path) -> dict[str, Any]:
    plan = load(path)
    base_item = plan.get("基础计划")
    if not base_item:
        return plan
    base_path = repo / str(base_item["path"])
    if not base_path.exists() or sha256(base_path) != base_item["sha256"]:
        raise RuntimeError("确认轮基础计划缺失或哈希不一致")
    base = load(base_path)
    override = plan.get("覆盖")
    if not isinstance(override, dict):
        raise RuntimeError("确认轮覆盖内容无效")
    merged = merge_dict(base, override)
    merged["基础计划"] = base_item
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load_registered_plan(args.plan, repo)
    if plan.get("status") != "frozen_before_route_stability_diagnostic":
        raise RuntimeError("逻辑专家分布稳定性诊断计划未冻结")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "status.json"

    def update(stage: str, **extra: Any) -> None:
        atomic_json(
            status_path,
            {
                "schema_version": "qtopomoe.phase8_route_stability_status.v1",
                "stage": stage,
                "updated_unix": time.time(),
                "supervisor_pid": os.getpid(),
                **extra,
            },
        )

    runtime = plan["运行环境"]
    candidate_path = Path(runtime["runtime_plan"])
    runtime_patch = repo / runtime["runtime_patch"]
    smoke_client = repo / "clients" / "smoke.py"
    input_checks = {}
    for name, expected_status in (
        ("capture_gate", "accepted"),
        ("candidate_generation_gate", "accepted"),
        ("online_abab_gate_readonly", "rejected"),
        ("quality_gate_readonly", "accepted"),
    ):
        item = plan["输入依据"][name]
        path = Path(item["path"])
        input_checks[name] = (
            path.exists()
            and sha256(path) == item["sha256"]
            and load(path).get("status") == expected_status
        )
    previous = plan["输入依据"].get("previous_diagnostic_readonly")
    if previous is not None:
        path = Path(previous["path"])
        input_checks["previous_diagnostic_readonly"] = (
            path.exists()
            and sha256(path) == previous["sha256"]
            and load(path).get("status") == previous["expected_status"]
        )
    file_checks = {
        "python": Path(runtime["python_bin"]).exists(),
        "model": Path(runtime["model"]).exists(),
        "candidate_plan": candidate_path.exists()
        and sha256(candidate_path) == runtime["runtime_plan_file_sha256"],
        "runtime_patch": runtime_patch.exists()
        and sha256(runtime_patch) == runtime["runtime_patch_sha256"],
        "smoke_client": smoke_client.exists()
        and sha256(smoke_client) == runtime["client_sha256"],
    }
    checks = {**input_checks, **file_checks}
    if not all(checks.values()):
        update("preflight_failed", checks=checks)
        return 2

    candidate = load(candidate_path)
    mapping = candidate["physical_to_logical_map"]
    if canonical_sha256(mapping) != runtime["runtime_map_sha256"]:
        raise RuntimeError("候选 map 哈希不一致")
    identity = [list(range(len(mapping[0]))) for _ in mapping]
    identity_hash = canonical_sha256(identity)
    if identity_hash != plan["候选"]["identity_map_sha256"]:
        raise RuntimeError("identity map 哈希不一致")

    diagnostic_path = root / "route_diagnostic_windows.jsonl"
    runtime["diagnostic_export_path"] = str(diagnostic_path)
    runtime["diagnostic_export_every"] = 1
    atomic_json(root / "diagnostic_plan.snapshot.json", plan)
    control_path = root / "placement_control.json"

    def control(generation: int, action: str, target: str) -> None:
        atomic_json(
            control_path,
            {
                "schema_version": "qtopomoe.placement_control.v1",
                "generation": generation,
                "action": action,
                "target_sha256": target,
            },
        )

    phase_ranges: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    manifest = None
    process = None
    rollback = None
    pipeline_error = None
    try:
        update("waiting_existing_service_idle", checks=checks)
        wait_service_idle(plan, update)
        manifest = capture_service(plan, root)
        update("stopping_existing_service", original_pid=manifest["original_pid"])
        stop_group(int(manifest["original_pgid"]))
        wait_gpus_idle(list(range(8)), 600)
        update("launching_diagnostic_service")
        process = start_canary(plan, repo, root)
        wait_canary_health(process, plan)
        for index, phase in enumerate(plan["执行顺序"]):
            generation = phase.get("generation")
            action = phase.get("action")
            if generation is not None and action is not None:
                target = runtime["runtime_map_sha256"] if action == "apply" else identity_hash
                control(int(generation), str(action), target)
            name = str(phase["phase"])
            arm_name = str(phase["arm"])
            phase_plan = json.loads(json.dumps(plan, ensure_ascii=False))
            phase_plan["负载"] = plan["诊断负载"][arm_name]["负载"]
            start_line = jsonl_rows(diagnostic_path)
            update("running_phase", index=index, phase=name, arm=arm_name)
            run_phase(phase_plan, repo, root, name)
            end_line = jsonl_rows(diagnostic_path)
            summaries[name] = load(root / name / "summary.json")
            phase_ranges.append(
                {
                    "phase": name,
                    "arm": arm_name,
                    "role": phase["role"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "generation": generation,
                    "action": action,
                }
            )
            atomic_json(
                root / "phase_ranges.json",
                {
                    "schema_version": "qtopomoe.phase8_route_stability_ranges.v1",
                    "phases": phase_ranges,
                },
            )
        update("diagnostic_workload_completed")
    except BaseException as error:
        pipeline_error = repr(error)
        update("diagnostic_failed", error=pipeline_error)
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
            update("restore_failed", pipeline_error=pipeline_error, error=repr(error))
            return 2

    if pipeline_error:
        result = {
            "schema_version": "qtopomoe.phase8_route_stability_gate.v1",
            "status": "invalid",
            "pipeline_error": pipeline_error,
            "input_checks": checks,
            "rollback": rollback,
        }
        atomic_json(root / "diagnostic_gate.json", result)
        update("completed", gate_status="invalid")
        return 1

    analysis = analyze(plan, root)
    log_text = (root / "server.log").read_text(encoding="utf-8", errors="replace")
    expected_commits = [
        (int(phase["generation"]), str(phase["action"]))
        for phase in plan["执行顺序"]
        if phase.get("generation") is not None
    ]
    operational_gates = {
        "input_checks": all(checks.values()),
        "all_requests_completed_clean": all(
            value.get("completed") == value.get("requests")
            and value.get("failed") == 0
            for value in summaries.values()
        ),
        "all_control_generations_committed_on_8_ranks": all(
            committed_ranks(log_text, generation, action) == list(range(8))
            for generation, action in expected_commits
        ),
        "analysis_classified": analysis["status"] == "classified",
        "original_service_restored": rollback is not None
        and rollback.get("health_http") == 200
        and rollback.get("gpu_ids") == [4, 5, 6, 7],
    }
    result = {
        "schema_version": "qtopomoe.phase8_route_stability_gate.v1",
        "status": "classified" if all(operational_gates.values()) else "invalid",
        "说明": "classified 表示成功定位成因，不表示候选获得部署准入。旧 A/B/A/B Gate 保持 rejected。",
        "operational_gates": operational_gates,
        "classification": analysis["classification"],
        "analysis": analysis,
        "runtime_commits": {
            f"generation_{generation}_{action}_ranks": committed_ranks(
                log_text, generation, action
            )
            for generation, action in expected_commits
        },
        "input_checks": checks,
        "rollback": rollback,
    }
    atomic_json(root / "route_stability_analysis.json", analysis)
    atomic_json(root / "diagnostic_gate.json", result)
    update("completed", gate_status=result["status"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "classified" else 1


if __name__ == "__main__":
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: (_ for _ in ()).throw(RuntimeError(f"signal {signum}")),
    )
    raise SystemExit(main())
