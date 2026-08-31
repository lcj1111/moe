#!/usr/bin/env python3
# 作用：判断重复观测是否满足 selector 校准的输入条件。
"""Audit whether repeated Phase 8 observations can enter selector calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector.kernel_db import KernelDatabase
from selector.strategy_selector import StrategyCandidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(candidates: Sequence[StrategyCandidate], observations: Sequence[Mapping[str, Any]],
          kernel_db: KernelDatabase) -> dict[str, Any]:
    required_buckets = sorted({
        int(bucket)
        for observation in observations
        for bucket in observation["real_M_hist"]
    })
    coverage = []
    for candidate in candidates:
        backend = candidate.cost_kernel_backend or candidate.kernel_backend
        missing_buckets = [
            bucket for bucket in required_buckets
            if kernel_db.best(bucket, candidate.quant_format, backend) is None
        ]
        missing_service_rows = [
            index for index, observation in enumerate(observations)
            if candidate.candidate_id not in observation.get("measured_p99_ms_by_candidate", {})
        ]
        missing_ci_rows = [
            index for index, observation in enumerate(observations)
            if candidate.candidate_id not in observation.get("placement", {}).get(
                "measured_p99_bootstrap_95ci_by_candidate", {})
        ]
        coverage.append({
            "candidate_id": candidate.candidate_id,
            "quant_format": candidate.quant_format,
            "runtime_kernel_backend": candidate.kernel_backend,
            "cost_kernel_backend": backend,
            "required_m_buckets": required_buckets,
            "missing_kernel_m_buckets": missing_buckets,
            "missing_service_observation_rows": missing_service_rows,
            "missing_bootstrap_ci_rows": missing_ci_rows,
            "calibration_ready": not (
                missing_buckets or missing_service_rows or missing_ci_rows),
        })
    blockers = [
        {
            "candidate_id": row["candidate_id"],
            "missing_kernel_m_buckets": row["missing_kernel_m_buckets"],
            "missing_service_observation_rows": row["missing_service_observation_rows"],
            "missing_bootstrap_ci_rows": row["missing_bootstrap_ci_rows"],
        }
        for row in coverage if not row["calibration_ready"]
    ]
    return {
        "status": "ready" if not blockers else "blocked_missing_calibration_inputs",
        "candidate_count": len(candidates),
        "observation_count": len(observations),
        "required_m_buckets": required_buckets,
        "candidate_coverage": coverage,
        "blockers": blockers,
        "gate": {
            "all_candidates_have_kernel_coverage": not any(
                row["missing_kernel_m_buckets"] for row in coverage),
            "all_candidates_have_service_observations": not any(
                row["missing_service_observation_rows"] for row in coverage),
            "all_candidates_have_bootstrap_ci": not any(
                row["missing_bootstrap_ci_rows"] for row in coverage),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--kernel-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = [
        StrategyCandidate(**row)
        for row in json.loads(args.candidates.read_text(encoding="utf-8"))
    ]
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    kernel_db = KernelDatabase.load(args.kernel_db)
    result = audit(candidates, observations, kernel_db)
    result.update({
        "schema_version": "qtopomoe.phase8_calibration_readiness.v1",
        "inputs": {
            "candidates": str(args.candidates),
            "candidates_sha256": _sha256(args.candidates),
            "observations": str(args.observations),
            "observations_sha256": _sha256(args.observations),
            "kernel_db": str(args.kernel_db),
            "kernel_db_sha256": _sha256(args.kernel_db),
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))


if __name__ == "__main__":
    main()
