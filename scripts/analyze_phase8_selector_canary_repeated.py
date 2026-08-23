#!/usr/bin/env python3
"""聚合 one-shot placement 五轮配对 canary，并生成机器可读 Gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    schedule = load(args.schedule)
    expected = schedule["rounds"]
    rounds: list[dict[str, Any]] = []
    for item in expected:
        number = int(item["round"])
        run_root = args.root / "runs" / f"round_{number:02d}"
        gate_path = run_root / "canary_gate.json"
        if not gate_path.exists():
            rounds.append({
                "round": number,
                "seed": int(item["seed"]),
                "completed": False,
                "status": "missing",
            })
            continue
        gate = load(gate_path)
        rollback = gate.get("rollback", {})
        rounds.append({
            "round": number,
            "seed": int(item["seed"]),
            "completed": True,
            "status": gate.get("status"),
            "gate_sha256": sha256(gate_path),
            "stable_p99_ms": gate["latency_ms"]["stable_p99"],
            "migration_p99_ms": gate["latency_ms"]["migration_p99"],
            "recovery_p99_ms": gate["latency_ms"]["recovery_p99"],
            "recovery_p99_ratio_to_stable": gate["latency_ms"][
                "recovery_p99_ratio_to_stable"
            ],
            "individual_gates": gate.get("gates", {}),
            "rollback_healthy": (
                rollback.get("health_http") == 200
                and rollback.get("process_alive") is True
                and rollback.get("command_matches") is True
                and rollback.get("gpu_ids") == [4, 5, 6, 7]
            ),
        })

    aggregate_gates = {
        "frozen_repeated_schedule": (
            schedule.get("status") == "frozen_before_repeated_canary"
        ),
        "all_5_rounds_completed": (
            len(rounds) == int(schedule["aggregate_gate"]["required_rounds"])
            and all(row["completed"] for row in rounds)
        ),
        "all_individual_gates_accepted": all(
            row["status"] == "accepted" for row in rounds
        ),
        "recovery_p99_within_105pct_each_round": all(
            row.get("recovery_p99_ratio_to_stable", float("inf"))
            <= float(
                schedule["aggregate_gate"][
                    "recovery_p99_ratio_to_stable_max_each_round"
                ]
            )
            for row in rounds
        ),
        "all_original_service_restores_healthy": all(
            row.get("rollback_healthy") is True for row in rounds
        ),
    }
    accepted = all(aggregate_gates.values())
    ratios = [
        float(row["recovery_p99_ratio_to_stable"])
        for row in rounds
        if "recovery_p99_ratio_to_stable" in row
    ]
    result = {
        "schema_version": "qtopomoe.phase8_selector_one_shot_repeated_gate.v1",
        "status": "accepted" if accepted else "rejected",
        "scope": "预注册五轮完整负载配对canary；不等同于自动闭环验收",
        "schedule": {
            "path": str(args.schedule),
            "sha256": sha256(args.schedule),
            "candidate_id": schedule["candidate_id"],
        },
        "rounds": rounds,
        "recovery_p99_ratio": {
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "mean": sum(ratios) / len(ratios) if ratios else None,
        },
        "gates": aggregate_gates,
        "后续边界": schedule["后续边界"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "gates": aggregate_gates}, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
