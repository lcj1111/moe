#!/usr/bin/env python3
"""按预注册种子顺序执行 one-shot placement 五轮配对 canary。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    schedule = load(args.schedule)
    if schedule.get("status") != "frozen_before_repeated_canary":
        raise RuntimeError("五轮调度未冻结")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "schedule.snapshot.json", schedule)
    base_path = repo / schedule["base_plan"]
    if not base_path.exists() or sha256(base_path) != schedule["base_plan_sha256"]:
        raise RuntimeError("五轮基础计划哈希不匹配")
    base = load(base_path)
    if base.get("status") != "frozen_before_canary":
        raise RuntimeError("五轮基础计划未冻结")
    plans = root / "plans"
    runs = root / "runs"
    plans.mkdir()
    runs.mkdir()
    status_path = root / "status.json"

    def update(stage: str, **extra: Any) -> None:
        atomic_json(status_path, {
            "schema_version": "qtopomoe.phase8_selector_one_shot_repeated_status.v1",
            "stage": stage,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            **extra,
        })

    completed_rounds: list[int] = []
    update("starting", completed_rounds=completed_rounds)
    for item in schedule["rounds"]:
        number = int(item["round"])
        seed = int(item["seed"])
        round_name = f"round_{number:02d}"
        run_root = runs / round_name
        plan_path = plans / f"{round_name}.json"
        plan = copy.deepcopy(base)
        plan["运行环境"]["output_root"] = str(run_root)
        plan["负载"]["seed"] = seed
        plan["重复轮次"] = {
            "schedule": str(args.schedule),
            "round": number,
            "seed": seed,
        }
        atomic_json(plan_path, plan)
        update(
            "running_round",
            current_round=number,
            current_seed=seed,
            completed_rounds=completed_rounds,
        )
        command = [
            base["运行环境"]["python_bin"],
            str(repo / "scripts" / "run_phase8_selector_limited_canary.py"),
            "--plan", str(plan_path),
            "--output-root", str(run_root),
        ]
        with (root / f"{round_name}.supervisor.log").open("wb") as log:
            completed = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        gate_path = run_root / "canary_gate.json"
        if not gate_path.exists():
            update(
                "stopped_missing_round_gate",
                current_round=number,
                returncode=completed.returncode,
                completed_rounds=completed_rounds,
            )
            return 2
        gate = load(gate_path)
        completed_rounds.append(number)
        failed_gates = [name for name, passed in gate["gates"].items() if not passed]
        update(
            "round_completed",
            current_round=number,
            round_status=gate["status"],
            failed_gates=failed_gates,
            completed_rounds=completed_rounds,
        )
        if failed_gates and failed_gates != ["recovery_p99_within_105pct_of_stable"]:
            update(
                "stopped_non_p99_gate_failure",
                current_round=number,
                failed_gates=failed_gates,
                completed_rounds=completed_rounds,
            )
            return 2

    update("aggregating", completed_rounds=completed_rounds)
    aggregate = repo / "scripts" / "analyze_phase8_selector_canary_repeated.py"
    command = [
        base["运行环境"]["python_bin"], str(aggregate),
        "--schedule", str(args.schedule),
        "--root", str(root),
        "--output", str(root / "repeated_gate.json"),
    ]
    completed = subprocess.run(command, cwd=repo)
    gate = load(root / "repeated_gate.json")
    update(
        "completed",
        completed_rounds=completed_rounds,
        gate_status=gate["status"],
        analyzer_returncode=completed.returncode,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
