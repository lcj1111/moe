#!/usr/bin/env python3
"""对新 placement 候选执行同进程暖态 A/B/A/B 验证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
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


def load_metrics_since(log_path: Path, offset: int) -> dict[str, float | int]:
    text = log_path.read_bytes()[offset:].decode("utf-8", errors="replace")
    rows = [(float(expert), float(rank)) for expert, rank in re.findall(
        r"Worker_TP0_EP0.*QTOPOMOE_EPLB_LOAD_WINDOW\].*"
        r"expert_cv=([0-9.]+) rank_cv=([0-9.]+)", text
    )]
    if not rows:
        raise RuntimeError("测量阶段未采集到 rank0 暖态负载窗口")
    return {
        "expert_load_cv_layer_median": statistics.median(row[0] for row in rows),
        "rank_load_cv": statistics.median(row[1] for row in rows),
        "samples": len(rows),
    }


def committed_ranks(text: str, generation: int, action: str) -> list[int]:
    return sorted({int(rank) for rank in re.findall(
        rf"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_CONTROL_COMMITTED\] "
        rf"generation={generation} action={action}", text
    )})


def request_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["request_id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load(args.plan)
    if plan.get("status") != "frozen_before_abab":
        raise RuntimeError("A/B/A/B 计划未冻结")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "status.json"

    def update(stage: str, **extra: Any) -> None:
        atomic_json(status_path, {
            "schema_version": "qtopomoe.phase8_warm_placement_abab_status.v1",
            "stage": stage,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            **extra,
        })

    runtime = plan["运行环境"]
    candidate_path = Path(runtime["runtime_plan"])
    capture_gate_path = Path(plan["准入依据"]["capture_gate"])
    generation_gate_path = Path(plan["准入依据"]["candidate_generation_gate"])
    candidate = load(candidate_path) if candidate_path.exists() else {}
    checks = {
        "capture_gate": capture_gate_path.exists()
        and sha256(capture_gate_path) == plan["准入依据"]["capture_gate_sha256"]
        and load(capture_gate_path).get("status") == "accepted",
        "candidate_generation_gate": generation_gate_path.exists()
        and sha256(generation_gate_path)
        == plan["准入依据"]["candidate_generation_gate_sha256"]
        and load(generation_gate_path).get("status") == "accepted",
        "candidate_file": candidate_path.exists()
        and sha256(candidate_path) == runtime["runtime_plan_file_sha256"],
        "candidate_id": candidate.get("candidate_id") == plan["候选"]["candidate_id"],
        "candidate_map_hash": candidate.get("physical_to_logical_map_sha256")
        == runtime["runtime_map_sha256"],
        "runtime_patch": sha256(repo / runtime["runtime_patch"])
        == runtime["runtime_patch_sha256"],
        "client": sha256(repo / "clients" / "smoke.py") == runtime["client_sha256"],
    }
    if not all(checks.values()):
        update("preflight_failed", checks=checks)
        return 2
    mapping = candidate["physical_to_logical_map"]
    identity = [list(range(len(mapping[0]))) for _ in mapping]
    identity_hash = canonical_sha256(identity)
    atomic_json(root / "abab_plan.snapshot.json", plan)

    manifest = None
    process = None
    pipeline_error = None
    rollback = None
    phases: dict[str, dict[str, Any]] = {}

    def workload(name: str, measured: bool) -> None:
        log_path = root / "server.log"
        offset = log_path.stat().st_size
        run_phase(plan, repo, root, name)
        row: dict[str, Any] = {"summary": load(root / name / "summary.json")}
        if measured:
            row["load_metrics"] = load_metrics_since(log_path, offset)
        phases[name] = row

    def control(generation: int, action: str, target: str) -> None:
        atomic_json(root / "placement_control.json", {
            "schema_version": "qtopomoe.placement_control.v1",
            "generation": generation,
            "action": action,
            "target_sha256": target,
        })

    try:
        update("waiting_existing_service_idle", checks=checks)
        wait_service_idle(plan, update)
        manifest = capture_service(plan, root)
        update("stopping_existing_service", original_pid=manifest["original_pid"])
        stop_group(int(manifest["original_pgid"]))
        wait_gpus_idle(list(range(8)), 600)
        update("launching_abab_service")
        process = start_canary(plan, repo, root)
        wait_canary_health(process, plan)

        update("identity_warmup")
        workload("identity_warmup", measured=False)
        update("identity_a1")
        workload("identity_a1", measured=True)
        control(1, "apply", runtime["runtime_map_sha256"])
        update("candidate_commit_warmup_1")
        workload("candidate_commit_warmup_1", measured=False)
        update("candidate_b1")
        workload("candidate_b1", measured=True)
        control(2, "rollback", identity_hash)
        update("identity_commit_warmup_2")
        workload("identity_commit_warmup_2", measured=False)
        update("identity_a2")
        workload("identity_a2", measured=True)
        control(3, "apply", runtime["runtime_map_sha256"])
        update("candidate_commit_warmup_2")
        workload("candidate_commit_warmup_2", measured=False)
        update("candidate_b2")
        workload("candidate_b2", measured=True)
        update("abab_workload_completed")
    except BaseException as error:
        pipeline_error = repr(error)
        update("abab_failed", error=pipeline_error)
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
            "schema_version": "qtopomoe.phase8_warm_placement_abab_gate.v1",
            "status": "rejected",
            "pipeline_error": pipeline_error,
            "input_gates": checks,
            "original_service_restored": rollback is not None
            and rollback.get("health_http") == 200,
            "rollback": rollback,
        }
        atomic_json(root / "abab_gate.json", result)
        update("completed", gate_status="rejected")
        return 1

    measured_names = ["identity_a1", "candidate_b1", "identity_a2", "candidate_b2"]
    request_sets = {
        name: request_rows(root / name / "requests.jsonl") for name in measured_names
    }
    ids = sorted(request_sets[measured_names[0]])
    prompt_match = all(
        [request_sets[name][item]["prompt_sha256"] for name in measured_names]
        == [request_sets[measured_names[0]][item]["prompt_sha256"]] * len(measured_names)
        for item in ids
    )
    response_match = all(
        [request_sets[name][item]["response_sha256"] for name in measured_names]
        == [request_sets[measured_names[0]][item]["response_sha256"]] * len(measured_names)
        for item in ids
    )
    identity_p99 = [float(phases[name]["summary"]["e2e_ms"]["p99"])
                    for name in ("identity_a1", "identity_a2")]
    candidate_p99 = [float(phases[name]["summary"]["e2e_ms"]["p99"])
                     for name in ("candidate_b1", "candidate_b2")]
    identity_rank_cv = [float(phases[name]["load_metrics"]["rank_load_cv"])
                        for name in ("identity_a1", "identity_a2")]
    candidate_rank_cv = [float(phases[name]["load_metrics"]["rank_load_cv"])
                         for name in ("candidate_b1", "candidate_b2")]
    identity_expert_cv = [float(phases[name]["load_metrics"]["expert_load_cv_layer_median"])
                          for name in ("identity_a1", "identity_a2")]
    candidate_expert_cv = [float(phases[name]["load_metrics"]["expert_load_cv_layer_median"])
                           for name in ("candidate_b1", "candidate_b2")]
    p99_ratio = statistics.median(candidate_p99) / statistics.median(identity_p99)
    rank_cv_ratio = statistics.median(candidate_rank_cv) / statistics.median(identity_rank_cv)
    expert_cv_ratio = statistics.median(candidate_expert_cv) / statistics.median(identity_expert_cv)
    log_text = (root / "server.log").read_text(encoding="utf-8", errors="replace")
    all_summaries = [value["summary"] for value in phases.values()]
    thresholds = plan["gates"]
    gates = {
        "input_gates_accepted": all(checks.values()),
        "all_requests_succeeded_with_fixed_output": all(
            value.get("failed") == 0
            and value.get("output_tokens_total")
            == value.get("requests") * value.get("output_tokens_requested")
            for value in all_summaries
        ),
        "matched_prompts": prompt_match,
        "exact_response_hash_match": response_match,
        "apply_rollback_apply_committed_on_all_8_ranks": (
            committed_ranks(log_text, 1, "apply") == list(range(8))
            and committed_ranks(log_text, 2, "rollback") == list(range(8))
            and committed_ranks(log_text, 3, "apply") == list(range(8))
        ),
        "candidate_p99_median_ratio_within_limit": p99_ratio
        <= float(thresholds["candidate_p99_median_ratio_max"]),
        "candidate_rank_cv_improved": rank_cv_ratio
        <= float(thresholds["candidate_rank_cv_ratio_max"]),
        "logical_expert_cv_stable": float(thresholds["logical_expert_cv_ratio_min"])
        <= expert_cv_ratio <= float(thresholds["logical_expert_cv_ratio_max"]),
        "original_service_restored": rollback is not None
        and rollback.get("health_http") == 200
        and rollback.get("gpu_ids") == [4, 5, 6, 7],
    }
    result = {
        "schema_version": "qtopomoe.phase8_warm_placement_abab_gate.v1",
        "status": "accepted" if all(gates.values()) else "rejected",
        "candidate_id": plan["候选"]["candidate_id"],
        "candidate_map_sha256": runtime["runtime_map_sha256"],
        "gates": gates,
        "metrics": {
            "identity_p99_ms": identity_p99,
            "candidate_p99_ms": candidate_p99,
            "candidate_to_identity_p99_median_ratio": p99_ratio,
            "identity_rank_cv": identity_rank_cv,
            "candidate_rank_cv": candidate_rank_cv,
            "candidate_to_identity_rank_cv_median_ratio": rank_cv_ratio,
            "identity_logical_expert_cv": identity_expert_cv,
            "candidate_logical_expert_cv": candidate_expert_cv,
            "candidate_to_identity_logical_expert_cv_median_ratio": expert_cv_ratio,
        },
        "runtime_commits": {
            "generation_1_apply_ranks": committed_ranks(log_text, 1, "apply"),
            "generation_2_rollback_ranks": committed_ranks(log_text, 2, "rollback"),
            "generation_3_apply_ranks": committed_ranks(log_text, 3, "apply"),
        },
        "rollback": rollback,
    }
    atomic_json(root / "abab_gate.json", result)
    update("completed", gate_status=result["status"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(
        RuntimeError(f"signal {signum}")
    ))
    raise SystemExit(main())
