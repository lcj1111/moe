import json
import tempfile
import unittest
from pathlib import Path

from phase4.workload.generate_m_buckets import build_records, bucket
from selector.backend_selector import BackendSelector
from selector.kernel_db import KernelDatabase, KernelMeasurement
from selector.strategy_selector import CostModel, StrategyCandidate, StrategySelector, WorkloadObservation


class SelectorTests(unittest.TestCase):
    def test_buckets_and_records_are_deterministic(self):
        self.assertEqual(bucket(9), 16)
        a, b = build_records(), build_records()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 90)
        self.assertEqual(a[0]["prefill_m_bucket"], 256)

    def test_backend_requires_measurement(self):
        db = KernelDatabase([KernelMeasurement("cutlass", {"tile_m": 64}, 128, p95_us=4, measured=True)])
        self.assertEqual(BackendSelector(db).select(128).backend, "cutlass")
        with self.assertRaises(Exception):
            BackendSelector(db).select(256)

    def test_strategy_filters_and_selects(self):
        db = KernelDatabase([
            KernelMeasurement("cutlass", {}, 128, precision="bf16", p95_us=10, measured=True),
            KernelMeasurement("triton", {}, 128, precision="bf16", p95_us=20, measured=True),
        ])
        candidates = [
            StrategyCandidate("good", "bf16", "ckpt", 4, 1, 1, "numa0", "none", 0, "cutlass"),
            StrategyCandidate("bad-quality", "bf16", "ckpt", 4, 1, 1, "numa0", "none", 0, "triton", quality_valid=False),
        ]
        model = CostModel(communication_us_per_gb_by_mapping={"numa0": 100.0})
        chosen, score, meta = StrategySelector(candidates, db, cost_model=model).select(WorkloadObservation({128: 1.0}))
        self.assertEqual(chosen.candidate_id, "good")
        self.assertEqual(score, 0.01)
        self.assertEqual(meta["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
