#!/usr/bin/env python3
# 作用：检查有限 canary 的请求、计划和恢复指标。
"""生成 Phase 8 selector 有限 canary 的机器可读 Gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--runtime-plan", required=True, type=Path)
    parser.add_argument("--canary-plan", required=True, type=Path)
    parser.add_argument("--rollback-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    canary = load(args.canary_plan)
    runtime = canary["运行环境"]
    placement = load(args.runtime_plan)
    rollback = load(args.rollback_status)
    activation = load(args.root / "activation_status.json")
    summaries = {
        phase: load(args.root / phase / "summary.json")
        for phase in canary["负载"]["phases"]
    }
    requests = {
        phase: load_jsonl(args.root / phase / "requests.jsonl")
        for phase in canary["负载"]["phases"]
    }
    identity_fields = (
        "request_id", "prompt_sha256", "seed", "input_tokens_requested",
        "input_tokens_actual", "output_tokens_requested",
    )
    request_identities = {
        phase: [tuple(row.get(field) for field in identity_fields) for row in rows]
        for phase, rows in requests.items()
    }
    matched_request_stream = all(
        rows == request_identities["stable"] for rows in request_identities.values()
    )
    log_bytes = args.server_log.read_bytes()
    log = log_bytes.decode("utf-8", errors="replace")
    activation_offset = int(activation["server_log_offset_before_activation"])
    before_activation = log_bytes[:activation_offset]
    after_activation = log_bytes[activation_offset:]

    expected_map_hash = runtime["runtime_map_sha256"]
    applied = [
        (int(rank), int(call), digest)
        for rank, call, digest in re.findall(
            r"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_PLAN_APPLIED\] "
            r"call=(\d+).*sha256=([0-9a-f]{64})",
            log,
        )
    ]
    first_call = min((call for _, call, _ in applied), default=None)
    first_rows = [row for row in applied if row[1] == first_call]
    first_ranks = sorted({rank for rank, _, _ in first_rows})
    first_hashes = sorted({digest for _, _, digest in first_rows})
    applied_per_rank = Counter(rank for rank, _, _ in applied)
    one_shot_commits = [
        (int(rank), int(call), digest)
        for rank, call, digest in re.findall(
            r"Worker_TP(\d+)_EP\d+.*QTOPOMOE_EPLB_ONE_SHOT_COMMITTED\] "
            r"call=(\d+).*sha256=([0-9a-f]{64})",
            log,
        )
    ]
    commit_ranks = sorted({rank for rank, _, _ in one_shot_commits})
    commit_hashes = sorted({digest for _, _, digest in one_shot_commits})
    native_migration = bool(re.search(r"Rearranged experts\s+in ([0-9.]+) s", log))

    failures = {phase: int(value.get("failed", -1)) for phase, value in summaries.items()}
    complete = {
        phase: int(value.get("completed", -1)) == int(value.get("requests", -2))
        for phase, value in summaries.items()
    }
    fixed_output_tokens = all(
        all(
            row.get("status") == "ok"
            and row.get("finish_reason") == "length"
            and int(row.get("output_tokens", -1)) == int(row.get("output_tokens_requested", -2))
            for row in rows
        )
        for rows in requests.values()
    )
    stable_p99 = float(summaries["stable"]["e2e_ms"]["p99"])
    migration_p99 = float(summaries["migration"]["e2e_ms"]["p99"])
    recovery_p99 = float(summaries["recovery"]["e2e_ms"]["p99"])
    recovery_ratio = recovery_p99 / stable_p99

    manage_existing_service = canary["既有服务"].get("manage_service", True)
    gates = {
        "frozen_canary_plan": canary.get("status") == "frozen_before_canary",
        "runtime_plan_file_hash_matches": (
            sha256(args.runtime_plan) == runtime["runtime_plan_file_sha256"]
        ),
        "runtime_map_hash_matches": (
            placement.get("physical_to_logical_map_sha256") == expected_map_hash
        ),
        "plan_hash_consistent_on_all_8_ranks": (
            first_ranks == list(range(8))
            and first_hashes == [expected_map_hash]
        ),
        "plan_not_applied_before_explicit_activation": (
            b"QTOPOMOE_EPLB_PLAN_APPLIED" not in before_activation
            and b"QTOPOMOE_EPLB_PLAN_APPLIED" in after_activation
            and activation.get("runtime_map_sha256") == expected_map_hash
        ),
        "one_shot_plan_applied_exactly_once_per_rank": (
            applied_per_rank == Counter({rank: 1 for rank in range(8)})
            and commit_ranks == list(range(8))
            and commit_hashes == [expected_map_hash]
        ),
        "native_migration_observed": native_migration,
        "all_requests_succeeded": (
            all(value == 0 for value in failures.values()) and all(complete.values())
        ),
        "matched_request_stream_across_phases": matched_request_stream,
        "fixed_output_tokens_across_phases": (
            canary["负载"].get("ignore_eos") is not True or fixed_output_tokens
        ),
        "recovery_p99_within_105pct_of_stable": recovery_ratio <= 1.05,
    }
    if manage_existing_service:
        gates["rollback_original_service_available"] = (
            rollback.get("health_http") == 200
            and rollback.get("process_alive") is True
            and rollback.get("command_matches") is True
            and rollback.get("gpu_ids") == [4, 5, 6, 7]
        )
    else:
        gates["existing_service_not_managed_as_authorized"] = (
            rollback.get("service_management") == "not_in_scope"
            and rollback.get("authorized") is True
        )
    accepted = all(gates.values())
    result = {
        "schema_version": "qtopomoe.phase8_selector_limited_canary_gate.v1",
        "status": "accepted" if accepted else "rejected",
        "scope": canary.get(
            "scope", "单次有限canary；不等同于自动闭环验收"
        ),
        "canary_plan": {
            "path": str(args.canary_plan),
            "sha256": sha256(args.canary_plan),
            "candidate_id": canary["canary目标"]["candidate_id"],
            "representative_workload_id": canary["canary目标"]["representative_workload_id"],
        },
        "runtime_plan": {
            "path": str(args.runtime_plan),
            "file_sha256": sha256(args.runtime_plan),
            "map_sha256": placement.get("physical_to_logical_map_sha256"),
            "first_applied_call": first_call,
            "first_applied_ranks": first_ranks,
            "first_applied_hashes": first_hashes,
            "activation": activation,
            "applied_count_per_rank": dict(sorted(applied_per_rank.items())),
            "one_shot_commit_ranks": commit_ranks,
            "one_shot_commit_hashes": commit_hashes,
        },
        "requests": {
            "failures": failures,
            "complete": complete,
            "total": sum(int(value["requests"]) for value in summaries.values()),
            "finish_reasons": {
                phase: value.get("finish_reasons", {}) for phase, value in summaries.items()
            },
            "matched_identity_fields": list(identity_fields),
            "matched_request_stream_across_phases": matched_request_stream,
            "ignore_eos": canary["负载"].get("ignore_eos", False),
            "fixed_output_tokens_across_phases": fixed_output_tokens,
        },
        "latency_ms": {
            "stable_p99": stable_p99,
            "migration_p99": migration_p99,
            "recovery_p99": recovery_p99,
            "recovery_p99_ratio_to_stable": recovery_ratio,
        },
        "rollback": rollback,
        "gates": gates,
        "后续边界": (
            "accepted后仅允许进入trigger、cooldown、rollback自动闭环的独立验收；"
            "rejected则停止上线并保留失败证据。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "gates": gates}, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
