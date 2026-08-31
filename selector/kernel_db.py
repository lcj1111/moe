# 作用：维护带版本和有效性标记的 MoE kernel 实测数据库。
"""Versioned MoE kernel measurement database.

Stores per-backend/per-M-bucket p50/p95 latencies measured on real GPUs
(``scripts/bench_moe_kernel.py``).  Only rows with ``measured=True`` and a
valid latency are selectable; no synthetic latency is ever injected.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class KernelMeasurement:
    backend: str
    kernel_config: dict[str, Any]
    m_bucket: int
    precision: str = "bf16"
    kernel_name: str | None = None
    device: str | None = None
    min_us: float | None = None
    max_us: float | None = None
    repeats: int | None = None
    p50_us: float | None = None
    p95_us: float | None = None
    measured: bool = False
    valid: bool = True
    source: str = "unmeasured"
    error: str | None = None

    @property
    def score_us(self) -> float | None:
        return self.p95_us if self.p95_us is not None else self.p50_us

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "KernelMeasurement":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in row.items() if k in allowed})


class KernelDatabase:
    """Versioned database; no synthetic latency is treated as a measurement."""
    schema_version = "qtopomoe.kernel_db.v1"

    def __init__(self, measurements: Iterable[KernelMeasurement] = ()):
        self.measurements = list(measurements)

    @classmethod
    def load(cls, path: str | Path) -> "KernelDatabase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        else:
            if data.get("schema_version") != cls.schema_version:
                raise ValueError(f"unsupported kernel DB schema: {data.get('schema_version')}")
            rows = data.get("measurements", [])
        return cls(KernelMeasurement.from_row(row) for row in rows)

    def dump(self, path: str | Path) -> None:
        payload = {"schema_version": self.schema_version,
                   "measurements": [asdict(row) for row in self.measurements]}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def candidates(self, m_bucket: int, precision: str, backend: str | None = None,
                   measured_only: bool = True) -> list[KernelMeasurement]:
        rows = [r for r in self.measurements if r.m_bucket == m_bucket and r.precision == precision
                and r.valid and (backend is None or r.backend == backend)
                and (r.measured or not measured_only) and r.score_us is not None]
        return sorted(rows, key=lambda r: (r.score_us, r.backend, json.dumps(r.kernel_config, sort_keys=True)))

    def best(self, m_bucket: int, precision: str, backend: str | None = None,
             measured_only: bool = True) -> KernelMeasurement | None:
        rows = self.candidates(m_bucket, precision, backend, measured_only)
        return rows[0] if rows else None
