#!/usr/bin/env python3
"""Fit and group-cross-validate a transparent Phase 8 service correction.

The base model remains the measured kernel + communication + imbalance +
migration model.  Calibration adds only a non-negative affine correction per
candidate.  Each workload family (W1/W2/W3/W4) is held out in turn, so no
held-out median or oracle choice is used to fit the fold that evaluates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector.kernel_db import KernelDatabase
from selector.strategy_selector import (
    CostModel,
    StrategyCandidate,
    StrategySelector,
    WorkloadObservation,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workload_id(observation: WorkloadObservation) -> str:
    value = str(observation.placement.get("workload_id", ""))
    if not value:
        raise ValueError("every calibration observation needs placement.workload_id")
    return value


def workload_family(observation: WorkloadObservation) -> str:
    return workload_id(observation).split("_", 1)[0]


def uncertainty_weight(observation: WorkloadObservation, candidate_id: str) -> float:
    """Inverse-variance weight from bootstrap CI with a 5% median floor."""
    median = float(observation.measured_p99_ms_by_candidate[candidate_id])
    intervals = observation.placement.get(
        "measured_p99_bootstrap_95ci_by_candidate", {}
    )
    interval = intervals.get(candidate_id)
    if not interval:
        raise ValueError(f"missing bootstrap interval for {candidate_id}")
    low = float(interval["low_ms"])
    high = float(interval["high_ms"])
    if high < low:
        raise ValueError(f"invalid bootstrap interval for {candidate_id}")
    approximate_sigma = (high - low) / 3.92
    sigma_floor = max(1e-6, 0.05 * median)
    sigma = max(approximate_sigma, sigma_floor)
    return 1.0 / (sigma * sigma)


def fit_nonnegative_affine(
    samples: Sequence[tuple[float, float, float]],
) -> dict[str, float | int]:
    if len(samples) < 2:
        raise ValueError("affine calibration requires at least two samples")
    weight_sum = sum(weight for _, _, weight in samples)
    if weight_sum <= 0:
        raise ValueError("calibration weights must be positive")
    mean_x = sum(weight * x for x, _, weight in samples) / weight_sum
    mean_y = sum(weight * y for _, y, weight in samples) / weight_sum
    variance_x = sum(weight * (x - mean_x) ** 2 for x, _, weight in samples)
    covariance = sum(
        weight * (x - mean_x) * (y - mean_y) for x, y, weight in samples
    )
    slope = max(0.0, covariance / variance_x) if variance_x > 1e-18 else 0.0
    intercept_ms = max(0.0, mean_y - slope * mean_x)
    weighted_mse = sum(
        weight * (intercept_ms + slope * x - y) ** 2
        for x, y, weight in samples
    ) / weight_sum
    return {
        "scale": slope,
        "intercept_ms": intercept_ms,
        "sample_count": len(samples),
        "weighted_train_rmse_ms": weighted_mse ** 0.5,
        "raw_prediction_min_ms": min(x for x, _, _ in samples),
        "raw_prediction_max_ms": max(x for x, _, _ in samples),
    }


def fit_coefficients(
    candidates: Sequence[StrategyCandidate],
    observations: Sequence[WorkloadObservation],
    kernel_db: KernelDatabase,
    raw_model: CostModel,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for candidate in candidates:
        samples = []
        for observation in observations:
            if candidate.candidate_id not in observation.measured_p99_ms_by_candidate:
                raise ValueError(
                    f"missing service median for {candidate.candidate_id} in "
                    f"{workload_id(observation)}"
                )
            raw = raw_model.predict_p99_ms(candidate, observation, kernel_db)
            if raw is None:
                raise ValueError(
                    f"missing raw cost for {candidate.candidate_id} in "
                    f"{workload_id(observation)}"
                )
            measured = float(
                observation.measured_p99_ms_by_candidate[candidate.candidate_id]
            )
            samples.append((
                raw,
                measured,
                uncertainty_weight(observation, candidate.candidate_id),
            ))
        result[candidate.candidate_id] = fit_nonnegative_affine(samples)
    return result


def corrected_model(
    raw_model: CostModel, coefficients: Mapping[str, Mapping[str, float | int]]
) -> CostModel:
    return CostModel(
        communication_us_per_gb=raw_model.communication_us_per_gb,
        communication_us_per_gb_by_mapping=raw_model.communication_us_per_gb_by_mapping,
        imbalance_us_per_unit=raw_model.imbalance_us_per_unit,
        migration_us_per_gb=raw_model.migration_us_per_gb,
        num_model_layers=raw_model.num_model_layers,
        service_scale_by_candidate={
            candidate_id: float(row["scale"])
            for candidate_id, row in coefficients.items()
        },
        service_intercept_ms_by_candidate={
            candidate_id: float(row["intercept_ms"])
            for candidate_id, row in coefficients.items()
        },
    )


def p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def summarize_rows(rows: Sequence[Mapping[str, Any]], candidate_count: int) -> dict[str, Any]:
    regrets = [float(row["regret_pct"]) for row in rows]
    overhead = [float(row["decision_overhead_pct_of_service_p99"]) for row in rows]
    median_regret = statistics.median(regrets) if regrets else None
    p95_regret = p95(regrets)
    overhead_p95 = p95(overhead)
    gates = {
        "median_regret_le_5pct": median_regret is not None and median_regret <= 5,
        "p95_regret_le_10pct": p95_regret is not None and p95_regret <= 10,
        "controller_overhead_lt_1pct": overhead_p95 is not None and overhead_p95 < 1,
    }
    return {
        "rows": list(rows),
        "top1_accuracy": (
            sum(float(row["regret_pct"]) == 0.0 for row in rows) / len(rows)
            if rows else None
        ),
        "median_regret_pct": median_regret,
        "p95_regret_pct": p95_regret,
        "decision_overhead_pct_p95": overhead_p95,
        "invalid_config_rate": (
            sum(int(row["invalid_count"]) for row in rows)
            / (len(rows) * candidate_count)
            if rows and candidate_count else 0.0
        ),
        "gates": gates,
    }


def grouped_cross_validation(
    candidates: Sequence[StrategyCandidate],
    observations: Sequence[WorkloadObservation],
    kernel_db: KernelDatabase,
    raw_model: CostModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = sorted({workload_family(observation) for observation in observations})
    if len(families) < 2:
        raise ValueError("grouped calibration requires at least two workload families")
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for held_out_family in families:
        train = [
            observation for observation in observations
            if workload_family(observation) != held_out_family
        ]
        test = [
            observation for observation in observations
            if workload_family(observation) == held_out_family
        ]
        coefficients = fit_coefficients(candidates, train, kernel_db, raw_model)
        selector = StrategySelector(
            candidates, kernel_db,
            cost_model=corrected_model(raw_model, coefficients),
        )
        folds.append({
            "held_out_family": held_out_family,
            "train_workload_ids": [workload_id(observation) for observation in train],
            "held_out_workload_ids": [workload_id(observation) for observation in test],
            "coefficients": coefficients,
        })
        for observation in test:
            chosen, predicted, meta = selector.select(observation)
            measured = {
                candidate.candidate_id: float(
                    observation.measured_p99_ms_by_candidate[candidate.candidate_id]
                )
                for candidate in candidates
            }
            oracle = min(measured.values())
            chosen_measured = measured[chosen.candidate_id]
            regret = (chosen_measured - oracle) / oracle * 100.0
            overhead_pct = meta["decision_overhead_ms"] / chosen_measured * 100.0
            rows.append({
                "workload_id": workload_id(observation),
                "held_out_family": held_out_family,
                "chosen": chosen.candidate_id,
                "predicted_p99_ms": predicted,
                "chosen_measured_p99_ms": chosen_measured,
                "oracle_candidate": min(measured, key=lambda key: (measured[key], key)),
                "oracle_p99_ms": oracle,
                "regret_pct": regret,
                "decision_overhead_pct_of_service_p99": overhead_pct,
                **meta,
            })
    return rows, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--kernel-db", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--cost-db", type=Path, required=True)
    parser.add_argument("--num-model-layers", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = [
        StrategyCandidate(**row)
        for row in json.loads(args.candidates.read_text(encoding="utf-8"))
    ]
    observations = [
        WorkloadObservation(**row)
        for row in json.loads(args.observations.read_text(encoding="utf-8"))
    ]
    kernel_db = KernelDatabase.load(args.kernel_db)
    cost_rows = json.loads(args.cost_db.read_text(encoding="utf-8"))
    mapping_rates = {
        row["mapping"]: row["median_effective_us_per_gb"]
        for row in cost_rows.get("mapping_summary", [])
    }
    raw_model = CostModel(
        communication_us_per_gb_by_mapping=mapping_rates,
        num_model_layers=args.num_model_layers,
    )
    held_out_rows, folds = grouped_cross_validation(
        candidates, observations, kernel_db, raw_model
    )
    evaluation = summarize_rows(held_out_rows, len(candidates))
    full_coefficients = fit_coefficients(
        candidates, observations, kernel_db, raw_model
    )
    uncalibrated = StrategySelector(
        candidates, kernel_db, cost_model=raw_model
    ).evaluate(observations)
    all_gates_pass = all(value is True for value in evaluation["gates"].values())
    result = {
        "schema_version": "qtopomoe.phase8_service_calibration.v1",
        "status": "accepted" if all_gates_pass else "gate_failed",
        "method": {
            "model": "per_candidate_nonnegative_affine",
            "base_features": [
                "measured_kernel_prefill_plus_output_tokens_times_decode",
                "num_model_layers",
                "measured_mapping_communication",
                "route_imbalance",
                "migration_bytes",
            ],
            "fit": "weighted_least_squares",
            "uncertainty_weight": (
                "inverse variance from bootstrap 95% CI with 5% median sigma floor"
            ),
            "validation": "leave_one_workload_family_out_W1_W2_W3_W4",
            "constraints": "scale>=0, intercept_ms>=0",
            "num_model_layers": args.num_model_layers,
            "leakage_control": (
                "held-out family medians and oracle labels are excluded from fold fit"
            ),
        },
        "inputs": {
            "candidates": str(args.candidates),
            "candidates_sha256": sha256(args.candidates),
            "kernel_db": str(args.kernel_db),
            "kernel_db_sha256": sha256(args.kernel_db),
            "observations": str(args.observations),
            "observations_sha256": sha256(args.observations),
            "communication_cost_db": str(args.cost_db),
            "communication_cost_db_sha256": sha256(args.cost_db),
        },
        "candidate_count": len(candidates),
        "observation_count": len(observations),
        "held_out_evaluation": evaluation,
        "folds": folds,
        "full_data_deployment_coefficients": full_coefficients,
        "uncalibrated_baseline": {
            key: uncalibrated[key]
            for key in (
                "top1_accuracy", "median_regret_pct", "p95_regret_pct",
                "decision_overhead_pct_p95", "invalid_config_rate", "gates",
            )
        },
        "formal_gate_ready": all_gates_pass,
        "caveat": (
            "This is grouped cross-validation over 12 measured workloads, not an "
            "independent external test set. Prefix-cache and open-loop arrival "
            "matrices remain separate Runbook Gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "formal_gate_ready": result["formal_gate_ready"],
        "top1_accuracy": evaluation["top1_accuracy"],
        "median_regret_pct": evaluation["median_regret_pct"],
        "p95_regret_pct": evaluation["p95_regret_pct"],
    }))


if __name__ == "__main__":
    main()
