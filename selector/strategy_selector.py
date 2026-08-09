"""Phase 8 joint strategy selector and transparent cost model.

Evaluates TP/DP/EP candidates against workload observations (real-M
histogram, communication bytes, route histogram) using measured kernel
latencies and mapping costs, then reports predicted p99, oracle regret and
controller overhead.  See ``docs/Q-TopoMoE_Phase4_Phase8_framework.md``.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .kernel_db import KernelDatabase


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    quant_format: str
    checkpoint: str
    tp: int
    dp: int
    ep: int
    gpu_mapping: str
    eplb_policy: str
    redundant_experts: int
    kernel_backend: str
    kernel_config: dict[str, Any] = field(default_factory=dict)
    cost_kernel_backend: str | None = None
    memory_required_gb: float | None = None
    quality_valid: bool = True
    supported: bool = True
    measured_p99_ms: float | None = None


@dataclass(frozen=True)
class WorkloadObservation:
    real_M_hist: Mapping[int | str, float]
    communication_bytes: float = 0.0
    route_hist: Mapping[str, float] = field(default_factory=dict)
    placement: Mapping[str, Any] = field(default_factory=dict)
    migration_bytes: float = 0.0
    measured_p99_ms_by_candidate: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CostModel:
    """Transparent Phase 8 model; all coefficients are explicit and tunable."""
    communication_us_per_gb: float = 1000.0
    communication_us_per_gb_by_mapping: Mapping[str, float] = field(default_factory=dict)
    imbalance_us_per_unit: float = 100.0
    migration_us_per_gb: float = 500.0

    def imbalance_penalty(self, route_hist: Mapping[str, float]) -> float:
        values = [float(v) for v in route_hist.values() if float(v) >= 0]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return max(values) - mean

    def predict_p99_ms(self, candidate: StrategyCandidate, obs: WorkloadObservation,
                       kernel_db: KernelDatabase) -> float | None:
        compute_us = 0.0
        cost_backend = candidate.cost_kernel_backend or candidate.kernel_backend
        for raw_bucket, weight in obs.real_M_hist.items():
            row = kernel_db.best(int(raw_bucket), candidate.quant_format, cost_backend)
            if row is None or row.score_us is None:
                return None
            compute_us += float(weight) * row.score_us
        mapping_rate = self.communication_us_per_gb_by_mapping.get(
            candidate.gpu_mapping, self.communication_us_per_gb)
        comm_us = obs.communication_bytes / 1e9 * mapping_rate
        imbalance_us = self.imbalance_penalty(obs.route_hist) * self.imbalance_us_per_unit
        migration_us = obs.migration_bytes / 1e9 * self.migration_us_per_gb
        return (compute_us + comm_us + imbalance_us + migration_us) / 1000.0


class StrategySelector:
    def __init__(self, candidates: Sequence[StrategyCandidate], kernel_db: KernelDatabase,
                 gpu_memory_gb: float | None = None, cost_model: CostModel | None = None):
        self.candidates = list(candidates)
        self.kernel_db = kernel_db
        self.gpu_memory_gb = gpu_memory_gb
        self.cost_model = cost_model or CostModel()

    def _valid(self, c: StrategyCandidate) -> bool:
        return c.supported and c.quality_valid and (self.gpu_memory_gb is None or
               c.memory_required_gb is None or c.memory_required_gb <= self.gpu_memory_gb)

    def select(self, obs: WorkloadObservation) -> tuple[StrategyCandidate, float, dict[str, Any]]:
        start = time.perf_counter()
        scored: list[tuple[float, StrategyCandidate]] = []
        invalid = 0
        for candidate in self.candidates:
            if not self._valid(candidate):
                invalid += 1
                continue
            score = self.cost_model.predict_p99_ms(candidate, obs, self.kernel_db)
            if score is None:
                invalid += 1
            else:
                scored.append((score, candidate))
        if not scored:
            raise ValueError("no valid strategy candidate for observation")
        score, chosen = min(scored, key=lambda x: (x[0], x[1].candidate_id))
        overhead_ms = (time.perf_counter() - start) * 1000
        return chosen, score, {"decision_overhead_ms": overhead_ms, "invalid_count": invalid,
                              "candidate_count": len(self.candidates)}

    def evaluate(self, observations: Sequence[WorkloadObservation]) -> dict[str, Any]:
        rows = []
        for obs in observations:
            chosen, predicted, meta = self.select(obs)
            measured_by_candidate = {
                c.candidate_id: float(obs.measured_p99_ms_by_candidate.get(
                    c.candidate_id, c.measured_p99_ms))
                for c in self.candidates
                if self._valid(c) and (
                    c.candidate_id in obs.measured_p99_ms_by_candidate
                    or c.measured_p99_ms is not None)
            }
            oracle = min(measured_by_candidate.values(), default=None)
            chosen_measured = measured_by_candidate.get(chosen.candidate_id)
            regret = (None if oracle in (None, 0) or chosen_measured is None else
                      (chosen_measured - oracle) / oracle * 100)
            overhead_denominator_ms = chosen_measured or predicted
            overhead_pct = (meta["decision_overhead_ms"] / overhead_denominator_ms * 100
                            if overhead_denominator_ms > 0 else None)
            rows.append({"chosen": chosen.candidate_id, "predicted_p99_ms": predicted,
                         "chosen_measured_p99_ms": chosen_measured,
                         "oracle_p99_ms": oracle, "regret_pct": regret,
                         "decision_overhead_denominator_ms": overhead_denominator_ms,
                         "decision_overhead_pct_of_service_p99": overhead_pct, **meta})
        regrets = [r["regret_pct"] for r in rows if r["regret_pct"] is not None]
        overhead = [r["decision_overhead_pct_of_service_p99"] for r in rows
                    if r["decision_overhead_pct_of_service_p99"] is not None]
        evaluated_rows = [r for r in rows if r["regret_pct"] is not None]
        return {"rows": rows, "top1_accuracy": (sum(r["regret_pct"] == 0 for r in evaluated_rows) /
                len(evaluated_rows) if evaluated_rows else None),
                "median_regret_pct": statistics.median(regrets) if regrets else None,
                "p95_regret_pct": statistics.quantiles(regrets, n=20, method="inclusive")[18] if len(regrets) >= 2 else (regrets[0] if regrets else None),
                "decision_overhead_pct_p95": statistics.quantiles(overhead, n=20, method="inclusive")[18] if len(overhead) >= 2 else (overhead[0] if overhead else None),
                "invalid_config_rate": sum(r["invalid_count"] for r in rows) / (len(rows) * len(self.candidates)) if rows and self.candidates else 0.0,
                "gates": {"median_regret_le_5pct": (statistics.median(regrets) <= 5 if regrets else None),
                          "p95_regret_le_10pct": ((statistics.quantiles(regrets, n=20, method="inclusive")[18] if len(regrets) >= 2 else (regrets[0] if regrets else 0)) <= 10 if regrets else None),
                          "controller_overhead_lt_1pct": (statistics.quantiles(overhead, n=20, method="inclusive")[18] if len(overhead) >= 2 else (overhead[0] if overhead else 0)) < 1 if overhead else None}}


def candidate_from_dict(data: Mapping[str, Any]) -> StrategyCandidate:
    return StrategyCandidate(**dict(data))


def dump_candidates(path: str, candidates: Sequence[StrategyCandidate]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(c) for c in candidates], handle, indent=2, sort_keys=True)
