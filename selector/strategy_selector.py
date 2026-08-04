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


@dataclass(frozen=True)
class CostModel:
    """Transparent Phase 8 model; all coefficients are explicit and tunable."""
    communication_us_per_gb: float = 1000.0
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
        for raw_bucket, weight in obs.real_M_hist.items():
            row = kernel_db.best(int(raw_bucket), candidate.quant_format, candidate.kernel_backend)
            if row is None or row.score_us is None:
                return None
            compute_us += float(weight) * row.score_us
        comm_us = obs.communication_bytes / 1e9 * self.communication_us_per_gb
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
            measured = [c for c in self.candidates if c.measured_p99_ms is not None]
            oracle = min((c.measured_p99_ms for c in measured), default=None)
            regret = None if oracle in (None, 0) else (predicted - oracle) / oracle * 100
            rows.append({"chosen": chosen.candidate_id, "predicted_p99_ms": predicted,
                         "oracle_p99_ms": oracle, "regret_pct": regret, **meta})
        regrets = [r["regret_pct"] for r in rows if r["regret_pct"] is not None]
        overhead = [r["decision_overhead_ms"] for r in rows]
        return {"rows": rows, "top1_accuracy": sum(r["regret_pct"] == 0 for r in rows) / len(rows) if rows else None,
                "median_regret_pct": statistics.median(regrets) if regrets else None,
                "p95_regret_pct": statistics.quantiles(regrets, n=20, method="inclusive")[18] if len(regrets) >= 2 else (regrets[0] if regrets else None),
                "decision_overhead_ms_p95": statistics.quantiles(overhead, n=20, method="inclusive")[18] if len(overhead) >= 2 else (overhead[0] if overhead else None),
                "invalid_config_rate": sum(r["invalid_count"] for r in rows) / (len(rows) * len(self.candidates)) if rows and self.candidates else 0.0,
                "gates": {"median_regret_le_5pct": (statistics.median(regrets) <= 5 if regrets else None),
                          "p95_regret_le_10pct": ((statistics.quantiles(regrets, n=20, method="inclusive")[18] if len(regrets) >= 2 else (regrets[0] if regrets else 0)) <= 10 if regrets else None),
                          "controller_overhead_lt_1pct": None}}


def candidate_from_dict(data: Mapping[str, Any]) -> StrategyCandidate:
    return StrategyCandidate(**dict(data))


def dump_candidates(path: str, candidates: Sequence[StrategyCandidate]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(c) for c in candidates], handle, indent=2, sort_keys=True)
