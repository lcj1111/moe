#!/usr/bin/env python3
"""对新 placement 候选执行 identity/candidate/identity 质量夹测。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import time
from collections import Counter, defaultdict
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


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def committed_ranks(text: str, generation: int, action: str) -> list[int]:
    return sorted({int(rank) for rank in re.findall(
        rf"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_CONTROL_COMMITTED\] "
        rf"generation={generation} action={action}", text
    )})


def phase_stats(
    rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [row for row in items if row.get("correct") is not None]
        correct = sum(row.get("correct") is True for row in scored)
        return {
            "records": len(items),
            "failed": sum(bool(row.get("error")) for row in items),
            "truncated": sum(bool(row.get("truncated")) for row in items),
            "extraction_failed": sum(
                not row.get("error") and not row.get("truncated")
                and row.get("prediction") is None for row in items
            ),
            "scored": len(scored),
            "correct": correct,
            "accuracy": correct / len(scored) if scored else None,
        }

    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = manifest[row["id"]]
        by_benchmark[source["benchmark"]].append(row)
        category = source.get("category") or source.get("subject") or source["benchmark"]
        by_category[f"{source['benchmark']}::{category}"].append(row)
    result = aggregate(rows)
    result["finish_reason_counts"] = dict(sorted(Counter(
        str(row.get("finish_reason")) for row in rows
    ).items()))
    result["benchmarks"] = {
        key: aggregate(value) for key, value in sorted(by_benchmark.items())
    }
    result["categories"] = {
        key: aggregate(value) for key, value in sorted(by_category.items())
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load(args.plan)
    if plan.get("status") != "frozen_before_quality_equivalence":
        raise RuntimeError("质量等价计划未冻结")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "status.json"

    def update(stage: str, **extra: Any) -> None:
        atomic_json(status_path, {
            "schema_version": "qtopomoe.phase8_warm_placement_quality_status.v1",
            "stage": stage,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            **extra,
        })

    runtime = plan["运行环境"]
    quality = plan["质量评测"]
    inputs = plan["输入依据"]
    candidate_path = Path(runtime["runtime_plan"])
    manifest_path = Path(quality["manifest"])
    candidate = load(candidate_path) if candidate_path.exists() else {}
    checks = {
        "capture_gate_accepted": Path(inputs["capture_gate"]).exists()
        and sha256(Path(inputs["capture_gate"])) == inputs["capture_gate_sha256"]
        and load(Path(inputs["capture_gate"])).get("status") == "accepted",
        "candidate_generation_gate_accepted": Path(inputs["candidate_generation_gate"]).exists()
        and sha256(Path(inputs["candidate_generation_gate"]))
        == inputs["candidate_generation_gate_sha256"]
        and load(Path(inputs["candidate_generation_gate"])).get("status") == "accepted",
        "online_abab_gate_preserved_readonly": Path(inputs["online_abab_gate_readonly"]).exists()
        and sha256(Path(inputs["online_abab_gate_readonly"]))
        == inputs["online_abab_gate_sha256"]
        and load(Path(inputs["online_abab_gate_readonly"])).get("status")
        == inputs["online_abab_expected_status"],
        "candidate_file": candidate_path.exists()
        and sha256(candidate_path) == runtime["runtime_plan_file_sha256"],
        "candidate_id": candidate.get("candidate_id") == plan["候选"]["candidate_id"],
        "candidate_map_hash": candidate.get("physical_to_logical_map_sha256")
        == runtime["runtime_map_sha256"],
        "runtime_patch": sha256(repo / runtime["runtime_patch"])
        == runtime["runtime_patch_sha256"],
        "quality_client": sha256(repo / "clients" / "quality_eval.py")
        == runtime["quality_client_sha256"],
        "smoke_client": sha256(repo / "clients" / "smoke.py")
        == runtime["smoke_client_sha256"],
        "quality_manifest": manifest_path.exists()
        and sha256(manifest_path) == quality["manifest_sha256"],
    }
    if "quality_gate_v1_readonly" in inputs:
        previous_gate = Path(inputs["quality_gate_v1_readonly"])
        checks["quality_gate_v1_preserved_readonly"] = (
            previous_gate.exists()
            and sha256(previous_gate) == inputs["quality_gate_v1_sha256"]
            and load(previous_gate).get("status")
            == inputs["quality_gate_v1_expected_status"]
        )
    if "nvfp4_aux_regression_gate_readonly" in inputs:
        regression_gate = Path(inputs["nvfp4_aux_regression_gate_readonly"])
        checks["nvfp4_aux_regression_gate_accepted"] = (
            regression_gate.exists()
            and sha256(regression_gate)
            == inputs["nvfp4_aux_regression_gate_sha256"]
            and load(regression_gate).get("status") == "accepted"
        )
    if not all(checks.values()):
        update("preflight_failed", checks=checks)
        return 2

    source_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    source_rows = [json.loads(line) for line in source_lines]
    selected_ids = quality.get("仅评测ID")
    if selected_ids is not None:
        selected_set = set(selected_ids)
        if len(selected_set) != len(selected_ids):
            update("preflight_failed", checks=checks, error="仅评测ID存在重复")
            return 2
        manifest_lines = [
            line for line, row in zip(source_lines, source_rows)
            if row["id"] in selected_set
        ]
        manifest_rows = [json.loads(line) for line in manifest_lines]
        if {row["id"] for row in manifest_rows} != selected_set:
            update("preflight_failed", checks=checks, error="仅评测ID不完整")
            return 2
    else:
        manifest_lines = source_lines
        manifest_rows = source_rows
    if len(manifest_rows) != int(quality["records"]):
        update("preflight_failed", checks=checks, manifest_records=len(manifest_rows))
        return 2
    effective_manifest_path = root / "quality_manifest.effective.jsonl"
    effective_manifest_path.write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    effective_manifest_sha256 = sha256(effective_manifest_path)
    manifest = {row["id"]: row for row in manifest_rows}
    mapping = candidate["physical_to_logical_map"]
    identity = [list(range(len(mapping[0]))) for _ in mapping]
    identity_hash = canonical_sha256(identity)
    atomic_json(root / "quality_plan.snapshot.json", plan)

    phases: dict[str, dict[str, Any]] = {}
    service_manifest = None
    process = None
    rollback = None
    pipeline_error = None

    def control(generation: int, action: str, target: str) -> None:
        atomic_json(root / "placement_control.json", {
            "schema_version": "qtopomoe.placement_control.v1",
            "generation": generation,
            "action": action,
            "target_sha256": target,
        })

    def warmup(name: str) -> None:
        run_phase(plan, repo, root, name)
        phases[name] = {"summary": load(root / name / "summary.json")}

    def quality_phase(name: str) -> None:
        phase_dir = root / name
        phase_dir.mkdir(parents=True, exist_ok=False)
        command = [
            runtime["python_bin"], str(repo / "clients" / "quality_eval.py"),
            "--base-url", f"http://{runtime['host']}:{runtime['port']}/v1",
            "--model", runtime["served_model_name"],
            "--manifest", str(effective_manifest_path),
            "--output", str(phase_dir / "quality.results.jsonl"),
            "--summary", str(phase_dir / "quality.summary.json"),
            "--concurrency", str(quality["concurrency"]),
            "--seed", str(quality["seed"]),
            "--timeout", str(quality["request_timeout_seconds"]),
            "--checkpoint-every", str(quality["checkpoint_every"]),
        ]
        with (phase_dir / "client.log").open("wb") as log:
            completed = subprocess.run(
                command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
            )
        rows = jsonl(phase_dir / "quality.results.jsonl")
        phases[name] = {
            "returncode": completed.returncode,
            "summary": load(phase_dir / "quality.summary.json"),
            "stats": phase_stats(rows, manifest),
        }
        if completed.returncode != 0:
            raise RuntimeError(f"{name}质量客户端失败: {completed.returncode}")

    try:
        update("waiting_existing_service_idle", checks=checks)
        wait_service_idle(plan, update)
        service_manifest = capture_service(plan, root)
        update("stopping_existing_service", original_pid=service_manifest["original_pid"])
        stop_group(int(service_manifest["original_pgid"]))
        wait_gpus_idle(list(range(8)), 600)
        update("launching_quality_service")
        process = start_canary(plan, repo, root)
        wait_canary_health(process, plan)

        update("identity_warmup")
        warmup("identity_warmup")
        update("identity_a1")
        quality_phase("identity_a1")
        control(1, "apply", runtime["runtime_map_sha256"])
        update("candidate_commit_warmup")
        warmup("candidate_commit_warmup")
        update("candidate_b")
        quality_phase("candidate_b")
        control(2, "rollback", identity_hash)
        update("identity_commit_warmup")
        warmup("identity_commit_warmup")
        update("identity_a2")
        quality_phase("identity_a2")
        update("quality_workload_completed")
    except BaseException as error:
        pipeline_error = repr(error)
        update("quality_failed", error=pipeline_error)
    finally:
        try:
            if process is not None:
                try:
                    pgid = os.getpgid(process.pid)
                except ProcessLookupError:
                    pgid = int(load(root / "canary_service_process.json")["pgid"])
                stop_group(pgid)
            if service_manifest is not None:
                update("restoring_existing_service", pipeline_error=pipeline_error)
                rollback = restore_service(service_manifest, plan, root)
        except BaseException as error:
            update("restore_failed", pipeline_error=pipeline_error, error=repr(error))
            return 2

    if pipeline_error:
        result = {
            "schema_version": "qtopomoe.phase8_warm_placement_quality_gate.v1",
            "status": "rejected",
            "pipeline_error": pipeline_error,
            "input_gates": checks,
            "phases": phases,
            "original_service_restored": rollback is not None
            and rollback.get("health_http") == 200,
            "rollback": rollback,
            "conclusion_boundary": plan["结论边界"],
        }
        atomic_json(root / "quality_equivalence_gate.json", result)
        update("completed", gate_status="rejected")
        return 1

    names = ["identity_a1", "candidate_b", "identity_a2"]
    rows_by_phase = {
        name: {row["id"]: row for row in jsonl(
            root / name / "quality.results.jsonl"
        )} for name in names
    }
    ids = sorted(manifest)
    identity_a1 = rows_by_phase["identity_a1"]
    candidate_b = rows_by_phase["candidate_b"]
    identity_a2 = rows_by_phase["identity_a2"]
    unique_regressions = [item for item in ids if
                          identity_a1[item].get("correct") is True
                          and identity_a2[item].get("correct") is True
                          and candidate_b[item].get("correct") is not True]
    stable_improvements = [item for item in ids if
                           identity_a1[item].get("correct") is not True
                           and identity_a2[item].get("correct") is not True
                           and candidate_b[item].get("correct") is True]
    prediction_match_a1 = sum(
        candidate_b[item].get("prediction") == identity_a1[item].get("prediction")
        for item in ids
    )
    prediction_match_a2 = sum(
        candidate_b[item].get("prediction") == identity_a2[item].get("prediction")
        for item in ids
    )
    identity_prediction_match = sum(
        identity_a1[item].get("prediction") == identity_a2[item].get("prediction")
        for item in ids
    )
    stats = {name: phases[name]["stats"] for name in names}
    threshold = plan["gates"]
    allowed_finish = set(threshold["允许的finish_reason"])
    complete_clean = all(
        stats[name]["records"] == int(threshold["每轮完成数"])
        and stats[name]["failed"] == int(threshold["每轮失败数"])
        and stats[name]["truncated"] == int(threshold["每轮截断数"])
        and stats[name]["extraction_failed"]
        == int(threshold["每轮答案抽取失败数"])
        and set(stats[name]["finish_reason_counts"]).issubset(allowed_finish)
        for name in names
    )
    overall_no_regression = stats["candidate_b"]["correct"] >= min(
        stats["identity_a1"]["correct"], stats["identity_a2"]["correct"]
    )
    benchmark_no_regression = all(
        stats["candidate_b"]["benchmarks"][key]["correct"] >= min(
            stats["identity_a1"]["benchmarks"][key]["correct"],
            stats["identity_a2"]["benchmarks"][key]["correct"],
        ) for key in stats["candidate_b"]["benchmarks"]
    )
    category_no_regression = all(
        stats["candidate_b"]["categories"][key]["correct"] >= min(
            stats["identity_a1"]["categories"][key]["correct"],
            stats["identity_a2"]["categories"][key]["correct"],
        ) for key in stats["candidate_b"]["categories"]
    )
    log_text = (root / "server.log").read_text(encoding="utf-8", errors="replace")
    apply_ranks = committed_ranks(log_text, 1, "apply")
    rollback_ranks = committed_ranks(log_text, 2, "rollback")
    gates = {
        "input_gates_accepted": all(checks.values()),
        "all_three_rounds_complete_clean": complete_clean,
        "candidate_overall_correct_not_below_identity_floor": overall_no_regression,
        "candidate_benchmark_correct_not_below_identity_floor": benchmark_no_regression,
        "candidate_category_correct_not_below_identity_floor": category_no_regression,
        "no_new_regression_when_both_identity_rounds_correct":
        len(unique_regressions)
        == int(threshold["两轮identity均正确但候选错误的新增退化数"]),
        "apply_and_rollback_committed_on_all_8_ranks":
        apply_ranks == list(range(8)) and rollback_ranks == list(range(8)),
        "original_service_restored": rollback is not None
        and rollback.get("health_http") == 200
        and rollback.get("gpu_ids") == [4, 5, 6, 7],
    }
    result = {
        "schema_version": "qtopomoe.phase8_warm_placement_quality_gate.v1",
        "status": "accepted" if all(gates.values()) else "rejected",
        "candidate_id": plan["候选"]["candidate_id"],
        "candidate_map_sha256": runtime["runtime_map_sha256"],
        "source_quality_manifest_sha256": quality["manifest_sha256"],
        "effective_quality_manifest_sha256": effective_manifest_sha256,
        "gates": gates,
        "phase_stats": stats,
        "paired_audit": {
            "records": len(ids),
            "identity_a1_vs_a2_prediction_matches": identity_prediction_match,
            "candidate_vs_identity_a1_prediction_matches": prediction_match_a1,
            "candidate_vs_identity_a2_prediction_matches": prediction_match_a2,
            "unique_regression_ids": unique_regressions,
            "stable_improvement_ids": stable_improvements,
        },
        "runtime_commits": {
            "generation_1_apply_ranks": apply_ranks,
            "generation_2_rollback_ranks": rollback_ranks,
        },
        "rollback": rollback,
        "conclusion_boundary": plan["结论边界"],
    }
    atomic_json(root / "quality_equivalence_gate.json", result)
    update("completed", gate_status=result["status"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(
        RuntimeError(f"signal {signum}")
    ))
    raise SystemExit(main())
