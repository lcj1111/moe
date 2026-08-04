#!/usr/bin/env python3
"""Offline Phase 8 replay; reports a clear blocked state until kernel data exists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python3 scripts/replay_phase8.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector.kernel_db import KernelDatabase
from selector.strategy_selector import StrategyCandidate, StrategySelector, WorkloadObservation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--kernel-db", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        selector = StrategySelector(candidates, db)
        result.update({"status": "completed", "evaluation": selector.evaluate(observations)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
