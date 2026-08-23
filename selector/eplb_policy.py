"""Topology- and quantization-aware expert placement policy.

This module intentionally stays above the serving runtime.  It produces an
auditable placement plan from measured inputs and implements the hysteresis
for an online controller; applying the plan to vLLM/SGLang is a separate gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence


class PlacementError(ValueError):
    """Raised when a placement is invalid or cannot fit in HBM."""


@dataclass(frozen=True)
class TopologyCostMatrix:
    """Measured point-to-point cost model.

    ``pair_us_per_gb`` accepts either ``"src-dst"`` or ``"dst-src"`` keys.
    Missing pairs use ``default_us_per_gb``.  NUMA membership is kept
    separately so predicted cross-NUMA bytes remain transparent.
    """

    gpus: tuple[int, ...]
    numa_by_gpu: Mapping[int, int]
    pair_us_per_gb: Mapping[str, float] = field(default_factory=dict)
    default_us_per_gb: float = 1000.0

    def __post_init__(self) -> None:
        if not self.gpus or len(set(self.gpus)) != len(self.gpus):
            raise PlacementError("gpus must be non-empty and unique")
        if any(gpu not in self.numa_by_gpu for gpu in self.gpus):
            raise PlacementError("numa_by_gpu must cover every GPU")

    def cost_us_per_gb(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        return float(self.pair_us_per_gb.get(
            f"{src}-{dst}", self.pair_us_per_gb.get(f"{dst}-{src}", self.default_us_per_gb)
        ))

    def cross_numa(self, src: int, dst: int) -> bool:
        return self.numa_by_gpu[src] != self.numa_by_gpu[dst]


@dataclass(frozen=True)
class QuantizedExpertProfile:
    layer_id: int
    expert_id: int
    token_load: float
    service_time_us_per_token: float
    size_bytes: int
    activation_bytes_per_token: int
    source_gpu_weights: Mapping[int, float] = field(default_factory=dict)
    current_gpus: tuple[int, ...] = ()
    quant_format: str = "unknown"

    @property
    def key(self) -> str:
        return f"{self.layer_id}:{self.expert_id}"

    @property
    def demand_us(self) -> float:
        return float(self.token_load) * float(self.service_time_us_per_token)

    def validate(self, topology: TopologyCostMatrix) -> None:
        if self.token_load < 0 or self.service_time_us_per_token < 0:
            raise PlacementError(f"negative load/service time for expert {self.key}")
        if self.size_bytes <= 0 or self.activation_bytes_per_token <= 0:
            raise PlacementError(f"non-positive byte size for expert {self.key}")
        if any(gpu not in topology.gpus for gpu in self.source_gpu_weights):
            raise PlacementError(f"unknown source GPU for expert {self.key}")
        if any(weight < 0 for weight in self.source_gpu_weights.values()):
            raise PlacementError(f"negative source weight for expert {self.key}")
        if any(gpu not in topology.gpus for gpu in self.current_gpus):
            raise PlacementError(f"unknown current GPU for expert {self.key}")


@dataclass(frozen=True)
class OfflineEPLBConfig:
    load_balance_weight: float = 1.0
    topology_weight: float = 1.0
    migration_weight: float = 1.0
    migration_us_per_gb: float = 500.0
    max_redundant_experts: int = 0
    replica_load_threshold: float = 1.5


@dataclass(frozen=True)
class PlacementPlan:
    expert_to_gpu: Mapping[str, tuple[int, ...]]
    replicas: Mapping[str, int]
    gpu_load_us: Mapping[int, float]
    gpu_memory_bytes: Mapping[int, int]
    predicted_cross_numa_bytes: float
    predicted_dispatch_cost_us: float
    predicted_p99_us: float
    migration_bytes: int
    model_trace_sha256: str
    quant_formats: tuple[str, ...]
    plan_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stable_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OfflineExpertPlacement:
    """Deterministic greedy placement with explicit topology/HBM penalties."""

    def __init__(
        self,
        topology: TopologyCostMatrix,
        gpu_capacity_bytes: Mapping[int, int],
        hbm_headroom_bytes: Mapping[int, int] | None = None,
        config: OfflineEPLBConfig | None = None,
    ) -> None:
        self.topology = topology
        self.gpu_capacity_bytes = {gpu: int(gpu_capacity_bytes[gpu]) for gpu in topology.gpus}
        self.hbm_headroom_bytes = {
            gpu: int((hbm_headroom_bytes or {}).get(gpu, 0)) for gpu in topology.gpus
        }
        self.config = config or OfflineEPLBConfig()
        for gpu in topology.gpus:
            if self.gpu_capacity_bytes[gpu] <= self.hbm_headroom_bytes[gpu]:
                raise PlacementError(f"no usable HBM on GPU {gpu}")

    def _source_weights(self, expert: QuantizedExpertProfile) -> dict[int, float]:
        raw = dict(expert.source_gpu_weights)
        if not raw:
            return {gpu: 1.0 / len(self.topology.gpus) for gpu in self.topology.gpus}
        total = sum(raw.values())
        if total <= 0:
            raise PlacementError(f"source weights sum to zero for expert {expert.key}")
        return {gpu: float(raw.get(gpu, 0.0)) / total for gpu in self.topology.gpus}

    def _dispatch_stats(
        self, expert: QuantizedExpertProfile, destinations: Sequence[int]
    ) -> tuple[float, float]:
        cross_numa_bytes = 0.0
        dispatch_cost_us = 0.0
        for src, weight in self._source_weights(expert).items():
            dst = min(destinations, key=lambda gpu: (self.topology.cost_us_per_gb(src, gpu), gpu))
            dispatched = float(expert.token_load) * expert.activation_bytes_per_token * weight
            dispatch_cost_us += dispatched / 1e9 * self.topology.cost_us_per_gb(src, dst)
            if self.topology.cross_numa(src, dst):
                cross_numa_bytes += dispatched
        return cross_numa_bytes, dispatch_cost_us

    def _replica_count(self, expert: QuantizedExpertProfile, target_gpu_load_us: float) -> int:
        if self.config.max_redundant_experts <= 0 or target_gpu_load_us <= 0:
            return 1
        threshold = target_gpu_load_us * self.config.replica_load_threshold
        needed = max(1, math.ceil(expert.demand_us / threshold))
        return min(1 + self.config.max_redundant_experts, needed, len(self.topology.gpus))

    def plan(
        self, experts: Sequence[QuantizedExpertProfile], model_trace_sha256: str
    ) -> PlacementPlan:
        if not experts:
            raise PlacementError("at least one expert is required")
        if len(model_trace_sha256) != 64:
            raise PlacementError("model_trace_sha256 must be a SHA-256 hex digest")
        for expert in experts:
            expert.validate(self.topology)
        if len({expert.key for expert in experts}) != len(experts):
            raise PlacementError("duplicate layer:expert keys")

        gpu_load = {gpu: 0.0 for gpu in self.topology.gpus}
        gpu_memory = {gpu: 0 for gpu in self.topology.gpus}
        placement: dict[str, tuple[int, ...]] = {}
        total_demand = sum(expert.demand_us for expert in experts)
        target_load = total_demand / len(self.topology.gpus)

        ordered = sorted(experts, key=lambda item: (-item.demand_us, item.layer_id, item.expert_id))
        for expert in ordered:
            replica_count = self._replica_count(expert, target_load)
            selected: list[int] = []
            share = expert.demand_us / replica_count
            for _ in range(replica_count):
                candidates: list[tuple[float, int]] = []
                for gpu in self.topology.gpus:
                    if gpu in selected:
                        continue
                    usable = self.gpu_capacity_bytes[gpu] - self.hbm_headroom_bytes[gpu]
                    if gpu_memory[gpu] + expert.size_bytes > usable:
                        continue
                    tentative_load = dict(gpu_load)
                    tentative_load[gpu] += share
                    ideal = max(target_load, 1e-9)
                    load_score = max(tentative_load.values()) / ideal
                    _, comm_us = self._dispatch_stats(expert, selected + [gpu])
                    migration_us = (0 if gpu in expert.current_gpus else expert.size_bytes) / 1e9 * self.config.migration_us_per_gb
                    objective = (
                        self.config.load_balance_weight * load_score
                        + self.config.topology_weight * comm_us / ideal
                        + self.config.migration_weight * migration_us / ideal
                    )
                    candidates.append((objective, gpu))
                if not candidates:
                    raise PlacementError(
                        f"expert {expert.key} cannot fit with {replica_count} copy/copies and reserved HBM"
                    )
                _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
                selected.append(chosen)
                gpu_load[chosen] += share
                gpu_memory[chosen] += expert.size_bytes
            placement[expert.key] = tuple(sorted(selected))

        cross_numa_bytes = 0.0
        dispatch_cost_us = 0.0
        migration_bytes = 0
        by_key = {expert.key: expert for expert in experts}
        for key, destinations in placement.items():
            expert = by_key[key]
            cross, cost = self._dispatch_stats(expert, destinations)
            cross_numa_bytes += cross
            dispatch_cost_us += cost
            migration_bytes += sum(
                expert.size_bytes for gpu in destinations if gpu not in expert.current_gpus
            )

        predicted_p99_us = max(gpu_load.values()) + dispatch_cost_us / len(self.topology.gpus)
        payload: dict[str, object] = {
            "expert_to_gpu": placement,
            "gpu_load_us": gpu_load,
            "gpu_memory_bytes": gpu_memory,
            "model_trace_sha256": model_trace_sha256,
        }
        return PlacementPlan(
            expert_to_gpu=placement,
            replicas={key: len(gpus) - 1 for key, gpus in placement.items()},
            gpu_load_us=gpu_load,
            gpu_memory_bytes=gpu_memory,
            predicted_cross_numa_bytes=cross_numa_bytes,
            predicted_dispatch_cost_us=dispatch_cost_us,
            predicted_p99_us=predicted_p99_us,
            migration_bytes=migration_bytes,
            model_trace_sha256=model_trace_sha256,
            quant_formats=tuple(sorted({expert.quant_format for expert in experts})),
            plan_sha256=_stable_hash(payload),
        )


@dataclass(frozen=True)
class OnlineEPLBConfig:
    window_ms: int = 500
    min_requests: int = 1000
    ema_alpha: float = 0.2
    trigger_load_cv: float = 0.25
    trigger_windows: int = 3
    min_benefit_fraction: float = 0.05
    min_benefit_cost_ratio: float = 2.0
    min_residency_windows: int = 10
    cooldown_windows: int = 10
    rollback_regression_fraction: float = 0.05
    rollback_windows: int = 3


@dataclass(frozen=True)
class ControllerDecision:
    action: str
    reason: str
    observed_cv: float
    ema_cv: float
    consecutive_trigger_windows: int
    residency_windows: int
    cooldown_remaining: int
    p99_ema_ms: float
    rollback_reference_p99_ms: float | None


class OnlineEPLBController:
    """Runbook hysteresis/controller state machine; it never mutates a runtime."""

    def __init__(self, config: OnlineEPLBConfig | None = None) -> None:
        self.config = config or OnlineEPLBConfig()
        self.ema_cv: float | None = None
        self.consecutive_trigger_windows = 0
        self.residency_windows = self.config.min_residency_windows
        self.cooldown_remaining = 0
        self.active_rebalance = False
        self.p99_ema_ms: float | None = None
        self.pre_rebalance_p99_ms: float | None = None
        self.regression_windows = 0

    @staticmethod
    def _cv(loads: Sequence[float]) -> float:
        if not loads:
            raise ValueError("loads cannot be empty")
        mean = statistics.fmean(loads)
        if mean <= 0:
            return 0.0
        return statistics.pstdev(loads) / mean

    def observe(
        self,
        loads: Sequence[float],
        requests: int,
        current_p99_ms: float,
        candidate_benefit_fraction: float = 0.0,
        candidate_benefit_us: float = 0.0,
        migration_cost_us: float = math.inf,
        elapsed_ms: float = 0.0,
    ) -> ControllerDecision:
        cv = self._cv(loads)
        return self.observe_cv(
            cv=cv,
            requests=requests,
            elapsed_ms=elapsed_ms,
            current_p99_ms=current_p99_ms,
            candidate_benefit_fraction=candidate_benefit_fraction,
            candidate_benefit_us=candidate_benefit_us,
            migration_cost_us=migration_cost_us,
        )

    def observe_cv(
        self,
        cv: float,
        requests: int,
        current_p99_ms: float,
        candidate_benefit_fraction: float = 0.0,
        candidate_benefit_us: float = 0.0,
        migration_cost_us: float = math.inf,
        elapsed_ms: float = 0.0,
    ) -> ControllerDecision:
        """Consume a measured expert-load CV from a completed control window."""
        if cv < 0 or elapsed_ms < 0:
            raise ValueError("cv and elapsed_ms must be non-negative")
        self.ema_cv = cv if self.ema_cv is None else (
            self.config.ema_alpha * cv + (1 - self.config.ema_alpha) * self.ema_cv
        )
        # 回滚基线必须来自迁移前的平滑值。若直接使用触发窗口的单点 p99，
        # 正常抖动会被误判为连续退化；迁移生效后保持该基线冻结。
        if not self.active_rebalance:
            self.p99_ema_ms = current_p99_ms if self.p99_ema_ms is None else (
                self.config.ema_alpha * current_p99_ms
                + (1 - self.config.ema_alpha) * self.p99_ema_ms
            )
        self.residency_windows += 1
        cooldown_active = self.cooldown_remaining > 0
        if cooldown_active:
            self.cooldown_remaining -= 1

        if self.active_rebalance and self.pre_rebalance_p99_ms is not None:
            regressed = current_p99_ms > self.pre_rebalance_p99_ms * (
                1 + self.config.rollback_regression_fraction
            )
            self.regression_windows = self.regression_windows + 1 if regressed else 0
            if self.regression_windows >= self.config.rollback_windows:
                self.active_rebalance = False
                self.regression_windows = 0
                self.residency_windows = 0
                self.cooldown_remaining = self.config.cooldown_windows
                return self._decision("rollback", "p99 regression persisted", cv)
            return self._decision("hold", "monitoring active rebalance", cv)

        eligible_window = (
            requests >= self.config.min_requests
            or elapsed_ms >= self.config.window_ms
        )
        above_threshold = eligible_window and self.ema_cv > self.config.trigger_load_cv
        self.consecutive_trigger_windows = self.consecutive_trigger_windows + 1 if above_threshold else 0
        if not eligible_window:
            return self._decision("hold", "insufficient requests in window", cv)
        if cooldown_active:
            return self._decision("hold", "controller cooldown", cv)
        if self.residency_windows < self.config.min_residency_windows:
            return self._decision("hold", "minimum residency not reached", cv)
        if self.consecutive_trigger_windows < self.config.trigger_windows:
            return self._decision("hold", "load CV trigger not persistent", cv)
        if candidate_benefit_fraction < self.config.min_benefit_fraction:
            return self._decision("hold", "predicted benefit below threshold", cv)
        ratio = candidate_benefit_us / migration_cost_us if migration_cost_us > 0 else math.inf
        if ratio < self.config.min_benefit_cost_ratio:
            return self._decision("hold", "benefit/cost ratio below threshold", cv)

        self.active_rebalance = True
        self.pre_rebalance_p99_ms = float(self.p99_ema_ms or current_p99_ms)
        self.regression_windows = 0
        self.consecutive_trigger_windows = 0
        self.residency_windows = 0
        self.cooldown_remaining = self.config.cooldown_windows
        return self._decision("rebalance", "all runbook gates passed", cv)

    def _decision(self, action: str, reason: str, cv: float) -> ControllerDecision:
        return ControllerDecision(
            action=action,
            reason=reason,
            observed_cv=cv,
            ema_cv=float(self.ema_cv or 0.0),
            consecutive_trigger_windows=self.consecutive_trigger_windows,
            residency_windows=self.residency_windows,
            cooldown_remaining=self.cooldown_remaining,
            p99_ema_ms=float(self.p99_ema_ms or 0.0),
            rollback_reference_p99_ms=self.pre_rebalance_p99_ms,
        )
