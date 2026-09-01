#!/usr/bin/env python3
# 作用：验收 trigger、cooldown、apply 和 rollback 在线闭环。
"""执行 Phase 8 trigger/cooldown/rollback 有限在线闭环验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_phase8_selector_limited_canary import (
    atomic_json, capture_service, health_status, load, restore_service, run_phase,
    sha256, start_canary, stop_group, wait_canary_health, wait_gpus_idle,
    wait_service_idle,
)
from selector.eplb_policy import OnlineEPLBConfig, OnlineEPLBController


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))]


def load_metrics_since(log_path: Path, offset: int) -> dict[str, float | int]:
    text = log_path.read_bytes()[offset:].decode("utf-8", errors="replace")
    rows = [(float(expert), float(rank)) for expert, rank in re.findall(
        r"Worker_TP0_EP0.*QTOPOMOE_EPLB_LOAD_WINDOW\].*"
        r"expert_cv=([0-9.]+) rank_cv=([0-9.]+)", text
    )]
    if not rows:
        raise RuntimeError("控制窗口未采集到rank0逐层专家负载CV")
    return {
        "expert_load_cv_layer_median": statistics.median(row[0] for row in rows),
        "rank_load_cv": statistics.median(row[1] for row in rows),
        "samples": len(rows),
    }


def committed_ranks(log: str, generation: int, action: str) -> list[int]:
    return sorted({int(rank) for rank in re.findall(
        rf"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_CONTROL_COMMITTED\] "
        rf"generation={generation} action={action}", log
    )})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    repo = REPO
    plan = load(args.plan)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "status.json"

    def update(stage: str, **extra: Any) -> None:
        atomic_json(status_path, {
            "schema_version": "qtopomoe.phase8_selector_closed_loop_status.v1",
            "stage": stage,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            **extra,
        })

    if plan.get("status") != "frozen_before_closed_loop":
        raise RuntimeError("闭环计划未冻结")
    repeated = Path(plan["准入依据"]["repeated_gate"])
    runtime = plan["运行环境"]
    checks = {
        "repeated_gate": repeated.exists()
        and sha256(repeated) == plan["准入依据"]["repeated_gate_sha256"]
        and load(repeated).get("status") == "accepted",
        "runtime_plan": Path(runtime["runtime_plan"]).exists()
        and sha256(Path(runtime["runtime_plan"])) == runtime["runtime_plan_file_sha256"],
        "runtime_patch": sha256(repo / runtime["runtime_patch"])
        == runtime["runtime_patch_sha256"],
    }
    if not all(checks.values()):
        update("preflight_failed", checks=checks)
        return 2
    atomic_json(root / "closed_loop_plan.snapshot.json", plan)
    placement = load(Path(runtime["runtime_plan"]))["physical_to_logical_map"]
    identity = [list(range(len(placement[0]))) for _ in placement]
    identity_hash = canonical_sha256(identity)
    controller_values = dict(plan["控制器"])
    inputs = {
        "candidate_benefit_fraction": controller_values.pop("candidate_benefit_fraction"),
        "candidate_benefit_us": controller_values.pop("candidate_benefit_us_over_10_windows"),
        "migration_cost_us": controller_values.pop("migration_cost_us"),
    }
    controller = OnlineEPLBController(OnlineEPLBConfig(**controller_values))
    decisions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    overhead_ms: list[float] = []
    process = None
    manifest = None
    pipeline_error = None
    rollback = None
    manage_existing_service = plan["既有服务"].get("manage_service", True)

    def workload(name: str) -> tuple[dict[str, Any], float]:
        log_path = root / "server.log"
        offset = log_path.stat().st_size
        run_phase(plan, repo, root, name)
        summary = load(root / name / "summary.json")
        metrics = load_metrics_since(log_path, offset)
        cv = metrics["expert_load_cv_layer_median"]
        windows.append({"name": name, "load_cv": cv, "load_metrics": metrics,
                        "summary": summary})
        return summary, cv

    def observe(name: str, summary: dict[str, Any], cv: float, p99: float) -> str:
        started = time.perf_counter()
        decision = controller.observe_cv(
            cv=cv,
            requests=int(summary["requests"]),
            elapsed_ms=float(summary["wall_time_s"]) * 1000,
            current_p99_ms=p99,
            **inputs,
        )
        elapsed = (time.perf_counter() - started) * 1000
        overhead_ms.append(elapsed)
        decisions.append({"window": name, "decision_overhead_ms": elapsed, **asdict(decision)})
        return decision.action

    try:
        if manage_existing_service:
            update("waiting_existing_service_idle", checks=checks)
            wait_service_idle(plan, update)
            manifest = capture_service(plan, root)
            update("stopping_existing_service", original_pid=manifest["original_pid"])
            stop_group(int(manifest["original_pgid"]))
        else:
            update("existing_service_not_managed_waiting_gpus_idle", checks=checks)
        wait_gpus_idle(list(range(8)), 600)
        update("launching_canary_service")
        process = start_canary(plan, repo, root)
        wait_canary_health(process, plan)
        update("baseline")
        baseline, _ = workload("baseline")
        baseline_p99 = float(baseline["e2e_ms"]["p99"])
        for index in range(1, 4):
            update("trigger_windows", window=index)
            summary, cv = workload(f"trigger_{index:02d}")
            action = observe(f"trigger_{index:02d}", summary, cv, float(summary["e2e_ms"]["p99"]))
            if (index < 3 and action != "hold") or (index == 3 and action != "rebalance"):
                raise RuntimeError(f"trigger窗口{index}动作异常: {action}")
        atomic_json(root / "placement_control.json", {
            "schema_version": "qtopomoe.placement_control.v1",
            "generation": 1, "action": "apply",
            "target_sha256": runtime["runtime_map_sha256"],
        })
        update("applying_plan")
        workload("apply_confirm")
        for index in range(1, 11):
            update("cooldown_windows", window=index)
            summary, cv = workload(f"cooldown_{index:02d}")
            if observe(f"cooldown_{index:02d}", summary, cv, float(summary["e2e_ms"]["p99"])) != "hold":
                raise RuntimeError(f"cooldown窗口{index}未保持hold")
        injected_p99 = controller.pre_rebalance_p99_ms * float(
            plan["故障注入"]["rollback_p99_multiplier"]
        )
        for index in range(1, 4):
            update("rollback_fault_windows", window=index)
            summary, cv = workload(f"rollback_fault_{index:02d}")
            action = observe(f"rollback_fault_{index:02d}", summary, cv, injected_p99)
            if (index < 3 and action != "hold") or (index == 3 and action != "rollback"):
                raise RuntimeError(f"rollback窗口{index}动作异常: {action}")
        atomic_json(root / "placement_control.json", {
            "schema_version": "qtopomoe.placement_control.v1",
            "generation": 2, "action": "rollback", "target_sha256": identity_hash,
        })
        update("applying_rollback")
        workload("rollback_confirm")
        update("post_rollback_recovery")
        recovery, _ = workload("post_rollback_recovery")
        update("workload_completed")
    except BaseException as error:
        pipeline_error = repr(error)
        update("pipeline_failed", error=pipeline_error)
    finally:
        try:
            if process is not None:
                try:
                    pgid = os.getpgid(process.pid)
                except ProcessLookupError:
                    pgid = int(load(root / "canary_service_process.json")["pgid"])
                stop_group(pgid)
            if manage_existing_service and manifest is not None:
                update("restoring_existing_service", pipeline_error=pipeline_error)
                rollback = restore_service(manifest, plan, root)
            elif not manage_existing_service:
                rollback = {
                    "schema_version": "qtopomoe.closed_loop_external_service_status.v1",
                    "service_management": "not_in_scope",
                    "authorized": True,
                    "说明": "按用户要求，本轮不管理或判定既有SGLang服务。",
                }
                atomic_json(root / "rollback_status.json", rollback)
                update("existing_service_not_managed", pipeline_error=pipeline_error)
        except BaseException as error:
            update("restore_failed", pipeline_error=pipeline_error, restore_error=repr(error))
            return 2
    if pipeline_error:
        restored = (
            rollback is not None
            and (
                (
                    manage_existing_service
                    and rollback.get("health_http") == 200
                    and rollback.get("gpu_ids") == [4, 5, 6, 7]
                )
                or (
                    not manage_existing_service
                    and rollback.get("service_management") == "not_in_scope"
                    and rollback.get("authorized") is True
                )
            )
        )
        failure = {
            "schema_version": "qtopomoe.phase8_selector_closed_loop_gate.v1",
            "status": "rejected",
            "pipeline_error": pipeline_error,
            "input_gates": checks,
            "decisions": decisions,
            "windows": [{
                "name": row["name"],
                "load_cv": row["load_cv"],
                "load_metrics": row["load_metrics"],
                "p99_ms": row["summary"]["e2e_ms"]["p99"],
            } for row in windows],
            "existing_service_handling_satisfied": restored,
            "rollback": rollback,
        }
        atomic_json(root / "closed_loop_gate.json", failure)
        update("completed", gate_status="rejected", pipeline_error=pipeline_error,
               existing_service_handling_satisfied=restored)
        return 1
    log = (root / "server.log").read_text(encoding="utf-8", errors="replace")
    recovery_p99 = float(recovery["e2e_ms"]["p99"])
    fixed_requests = all(
        value["summary"].get("failed") == 0
        and value["summary"].get("output_tokens_total")
        == value["summary"].get("requests") * value["summary"].get("output_tokens_requested")
        for value in windows
    )
    cooldown_decisions = [row for row in decisions if row["window"].startswith("cooldown_")]
    fault_decisions = [row for row in decisions if row["window"].startswith("rollback_fault_")]
    overhead_pct = [100 * value / (float(window["summary"]["wall_time_s"]) * 1000)
                    for value, window in zip(overhead_ms, [w for w in windows if w["name"] != "baseline" and w["name"] not in {"apply_confirm", "rollback_confirm", "post_rollback_recovery"}])]
    gates = {
        "repeated_canary_accepted": checks["repeated_gate"],
        "trigger_after_exactly_3_eligible_high_cv_windows": (
            [row["action"] for row in decisions[:3]] == ["hold", "hold", "rebalance"]
            and all(row["observed_cv"] > controller.config.trigger_load_cv for row in decisions[:3])
        ),
        "apply_committed_once_on_all_8_ranks": committed_ranks(log, 1, "apply") == list(range(8)),
        "cooldown_holds_for_10_windows": (
            len(cooldown_decisions) == 10
            and all(row["action"] == "hold" for row in cooldown_decisions)
            and cooldown_decisions[-1]["cooldown_remaining"] == 0
        ),
        "rollback_after_exactly_3_regression_windows": (
            [row["action"] for row in fault_decisions] == ["hold", "hold", "rollback"]
        ),
        "rollback_committed_once_on_all_8_ranks": committed_ranks(log, 2, "rollback") == list(range(8)),
        "decision_overhead_pct_p95_lt_1": percentile(overhead_pct, 0.95) < 1.0,
        "all_requests_succeeded_with_fixed_output": fixed_requests,
        "post_rollback_p99_within_105pct_of_baseline": recovery_p99 / baseline_p99 <= 1.05,
    }
    if manage_existing_service:
        gates["original_service_restored"] = (
            rollback is not None
            and rollback.get("health_http") == 200
            and rollback.get("gpu_ids") == [4, 5, 6, 7]
        )
    else:
        gates["existing_service_not_managed_as_authorized"] = (
            rollback is not None
            and rollback.get("service_management") == "not_in_scope"
            and rollback.get("authorized") is True
        )
    result = {
        "schema_version": "qtopomoe.phase8_selector_closed_loop_gate.v1",
        "status": "accepted" if all(gates.values()) else "rejected",
        "gates": gates,
        "controller_config": asdict(controller.config),
        "decisions": decisions,
        "load_cv_definition": "各MoE层逻辑专家负载CV的窗口中位数；rank CV仅作旁证",
        "windows": [{"name": row["name"], "load_cv": row["load_cv"],
                     "load_metrics": row["load_metrics"],
                     "p99_ms": row["summary"]["e2e_ms"]["p99"]} for row in windows],
        "fault_injection": plan["故障注入"],
        "runtime": {"apply_ranks": committed_ranks(log, 1, "apply"),
                    "rollback_ranks": committed_ranks(log, 2, "rollback"),
                    "placement_hash": runtime["runtime_map_sha256"],
                    "rollback_hash": identity_hash},
        "latency_ms": {"baseline_p99": baseline_p99, "post_rollback_p99": recovery_p99,
                       "ratio": recovery_p99 / baseline_p99},
        "decision_overhead_pct_p95": percentile(overhead_pct, 0.95),
        "rollback": rollback,
    }
    atomic_json(root / "closed_loop_gate.json", result)
    update("completed", gate_status=result["status"])
    print(json.dumps({"status": result["status"], "gates": gates}, ensure_ascii=False))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(RuntimeError(f"signal {signum}")))
    raise SystemExit(main())
