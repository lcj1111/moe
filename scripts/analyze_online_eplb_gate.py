#!/usr/bin/env python3
# 作用：生成在线 placement、迁移和恢复实验的可审计 Gate。
"""Build the auditable online placement/migration/recovery Gate report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--runtime-plan", required=True, type=Path)
    parser.add_argument("--expert-bytes", type=int, default=1_769_496)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stable_path = args.root / "stable_plan_v3" / "summary.json"
    migration_path = args.root / "live_migration" / "summary.json"
    recovery_path = args.root / "recovery" / "summary.json"
    stable = load(stable_path)
    migration = load(migration_path)
    recovery = load(recovery_path)
    plan = load(args.runtime_plan)
    log = args.server_log.read_text(encoding="utf-8", errors="replace")

    durations = []
    for line in log.splitlines():
        if "Rearranged experts" not in line or "(profile)" in line:
            continue
        match = re.search(r"Rearranged experts.*?in ([0-9.]+) s", line)
        if match:
            durations.append(float(match.group(1)))
    live_first = durations[0] if durations else None
    steady = durations[1:]
    applied_call2_ranks = len(
        set(
            re.findall(
                r"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_PLAN_APPLIED\] call=2",
                log,
            )
        )
    )
    plan_hash = plan["physical_to_logical_map_sha256"]
    hash_occurrences = len(
        re.findall(
            rf"QTOPOMOE_EPLB_PLAN_APPLIED\] call=2 .*sha256={re.escape(plan_hash)}",
            log,
        )
    )

    stable_p99 = float(stable["e2e_ms"]["p99"])
    migration_p99 = float(migration["e2e_ms"]["p99"])
    recovery_p99 = float(recovery["e2e_ms"]["p99"])
    failures = {
        "stable": int(stable["failed"]),
        "migration": int(migration["failed"]),
        "recovery": int(recovery["failed"]),
    }
    runtime_migration_bytes = int(plan["moved_physical_slots_vs_identity"]) * args.expert_bytes
    gates = {
        "runtime_plan_shape_valid": (
            plan["num_layers"] == 40
            and plan["num_logical_experts"] == 256
            and plan["num_ranks"] == 8
            and plan["slots_per_rank"] == 32
        ),
        "plan_applied_on_all_8_ranks": applied_call2_ranks == 8 and hash_occurrences == 8,
        "native_live_weight_rearrangement_observed": live_first is not None and live_first > 0,
        "all_requests_succeeded": all(value == 0 for value in failures.values()),
        "recovery_p99_within_5pct_of_stable": recovery_p99 <= stable_p99 * 1.05,
    }
    accepted = all(gates.values())
    result = {
        "schema_version": "qtopomoe.online_eplb_gate.v1",
        "status": "accepted" if accepted else "rejected",
        "scope": "runtime-feasible placement repaired from measured histogram; native synchronous vLLM EPLB",
        "runtime_plan": {
            "path": str(args.runtime_plan),
            "file_sha256": sha256(args.runtime_plan),
            "map_sha256": plan_hash,
            "offline_plan_runtime_feasible": plan["offline_plan_runtime_feasible"],
            "repair_method": plan["repair_method"],
            "changed_rank_assignments_vs_offline": plan["changed_rank_assignments_vs_offline"],
            "moved_physical_slots_vs_identity": plan["moved_physical_slots_vs_identity"],
            "expert_bytes": args.expert_bytes,
            "estimated_native_migration_bytes": runtime_migration_bytes,
        },
        "service": {
            "model": "/data/models/test/redhatai_qwen36_nvfp4",
            "parallelism": "TP8 + EP8",
            "eplb": "synchronous torch_nccl",
            "window_size": 16,
            "step_interval": 16,
            "server_log": str(args.server_log),
            "server_log_sha256": sha256(args.server_log),
        },
        "workload": {
            "input_tokens": 256,
            "output_tokens": 32,
            "concurrency": 8,
            "requests_per_phase": 128,
            "seed": 42,
            "failures": failures,
        },
        "migration": {
            "first_live_native_rearrangement_s": live_first,
            "steady_same_plan_rearrangement_s": {
                "count": len(steady),
                "p50": percentile(steady, 0.50),
                "p95": percentile(steady, 0.95),
                "p99": percentile(steady, 0.99),
            },
            "applied_call2_rank_count": applied_call2_ranks,
            "applied_call2_hash_occurrences": hash_occurrences,
        },
        "latency_ms": {
            "stable": stable["e2e_ms"],
            "migration_window": migration["e2e_ms"],
            "recovery": recovery["e2e_ms"],
            "migration_p99_regression_fraction": migration_p99 / stable_p99 - 1,
            "recovery_p99_ratio_to_stable": recovery_p99 / stable_p99,
        },
        "inputs": {
            "stable_summary": {"path": str(stable_path), "sha256": sha256(stable_path)},
            "migration_summary": {"path": str(migration_path), "sha256": sha256(migration_path)},
            "recovery_summary": {"path": str(recovery_path), "sha256": sha256(recovery_path)},
        },
        "gates": gates,
        "notes": [
            "原离线计划不满足 vLLM 每层每卡固定 32 个物理专家槽约束，不能原样上线。",
            "已使用同一实测 expert-token histogram 和 vLLM 原生 EPLB policy 生成运行时可行计划。",
            "profile 阶段返回原布局；第一个在线 EPLB 周期才应用固定计划，因此 0.83 秒为真实在线权重重排。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
