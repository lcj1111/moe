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
        self.assertEqual(len(a), 108)
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

    def test_regret_uses_measured_latency_of_selected_candidate(self):
        db = KernelDatabase([
            KernelMeasurement("proxy-a", {}, 128, precision="nvfp4", p95_us=10, measured=True),
            KernelMeasurement("proxy-b", {}, 128, precision="nvfp4", p95_us=20, measured=True),
        ])
        candidates = [
            StrategyCandidate("a", "nvfp4", "ckpt", 4, 1, 4, "numa0", "static", 0, "marlin",
                              cost_kernel_backend="proxy-a"),
            StrategyCandidate("b", "nvfp4", "ckpt", 8, 1, 8, "sys", "static", 0, "marlin",
                              cost_kernel_backend="proxy-b"),
        ]
        obs = WorkloadObservation(
            {128: 1.0},
            measured_p99_ms_by_candidate={"a": 120.0, "b": 100.0},
        )
        result = StrategySelector(candidates, db).evaluate([obs])
        row = result["rows"][0]
        self.assertEqual(row["chosen"], "a")
        self.assertEqual(row["chosen_measured_p99_ms"], 120.0)
        self.assertEqual(row["oracle_p99_ms"], 100.0)
        self.assertAlmostEqual(row["regret_pct"], 20.0)
        self.assertEqual(row["decision_overhead_denominator_ms"], 120.0)
        self.assertEqual(result["top1_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
