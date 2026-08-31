#!/usr/bin/env python3
# 作用：离线回放 Phase 8 成本模型并显式报告缺失输入。
"""Offline Phase 8 replay; reports a clear blocked state until kernel data exists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python3 scripts/replay_phase8.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector.kernel_db import KernelDatabase
from selector.strategy_selector import (
    CostModel,
    StrategyCandidate,
    StrategySelector,
    WorkloadObservation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--kernel-db", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-db", type=Path,
                        default=Path("configs/communication/nccl_cost_db.json"))
    args = parser.parse_args()
    candidates = [StrategyCandidate(**row) for row in json.loads(args.candidates.read_text(encoding="utf-8"))]
    observations = [WorkloadObservation(**row) for row in json.loads(args.observations.read_text(encoding="utf-8"))]
    db = KernelDatabase.load(args.kernel_db)
    result = {"schema_version": "qtopomoe.phase8_replay.v1", "candidate_count": len(candidates),
              "observation_count": len(observations), "kernel_measurement_count": len(db.measurements)}
    if not db.measurements:
        result.update({"status": "blocked_missing_kernel_measurements", "rows": [],
                       "reason": "Phase 4 kernel DB is a checked-in empty template; real CUDA measurements are required."})
    else:
        cost_rows = json.loads(args.cost_db.read_text(encoding="utf-8"))
        mapping_rates = {
            m["mapping"]: m["median_effective_us_per_gb"]
            for m in cost_rows.get("mapping_summary", [])
        }
        cost_model = CostModel(communication_us_per_gb_by_mapping=mapping_rates)
        selector = StrategySelector(candidates, db, cost_model=cost_model)
        evaluation = selector.evaluate(observations)
        evaluated_count = sum(
            row.get("regret_pct") is not None for row in evaluation["rows"])
        numeric_gates_pass = all(value is True for value in evaluation["gates"].values())
        formal_gate_ready = evaluated_count >= 4 and numeric_gates_pass
        result.update({
            "status": "completed" if formal_gate_ready else "completed_provisional",
            "evaluation": evaluation,
            "evaluated_observation_count": evaluated_count,
            "formal_gate_ready": formal_gate_ready,
            "formal_gate_requirement": (
                "at least four measured workload observations plus all numeric gates"),
        })
        result["communication_cost_source"] = str(args.cost_db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
