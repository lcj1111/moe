from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kernel_db import KernelDatabase, KernelMeasurement


class SelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    kernel_config: dict[str, Any]
    m_bucket: int
    precision: str
    predicted_us: float
    measured: bool
    source: str


class BackendSelector:
    """Select the lowest measured p95 kernel for an M bucket with safe fallback."""

    def __init__(self, database: KernelDatabase, allow_unmeasured: bool = False):
        self.database = database
        self.allow_unmeasured = allow_unmeasured

    def select(self, m_bucket: int, precision: str = "bf16", backend: str | None = None) -> BackendDecision:
        row: KernelMeasurement | None = self.database.best(
            m_bucket, precision, backend, measured_only=not self.allow_unmeasured)
        if row is None:
            raise SelectionError(f"no {'measured ' if not self.allow_unmeasured else ''}valid kernel for M={m_bucket} precision={precision}")
        assert row.score_us is not None
        return BackendDecision(row.backend, row.kernel_config, row.m_bucket, row.precision,
                               row.score_us, row.measured, row.source)
