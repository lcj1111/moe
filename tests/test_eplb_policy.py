# 作用：验证离线放置、在线迟滞和回滚策略。
import unittest

from selector.eplb_policy import (
    OfflineEPLBConfig,
    OfflineExpertPlacement,
    OnlineEPLBController,
    PlacementError,
    QuantizedExpertProfile,
    TopologyCostMatrix,
)


TRACE_SHA = "a" * 64


def topology():
    return TopologyCostMatrix(
        gpus=(0, 1, 2, 3),
        numa_by_gpu={0: 0, 1: 0, 2: 1, 3: 1},
        pair_us_per_gb={"0-1": 100, "2-3": 100, "0-2": 900, "0-3": 900,
                        "1-2": 900, "1-3": 900},
        default_us_per_gb=900,
    )


class OfflinePlacementTests(unittest.TestCase):
    def test_topology_locality_is_used(self):
        planner = OfflineExpertPlacement(topology(), {gpu: 10_000 for gpu in range(4)})
        experts = [
            QuantizedExpertProfile(0, 0, 100, 1, 1000, 16, {0: 1.0}, quant_format="nvfp4"),
            QuantizedExpertProfile(0, 1, 90, 1, 1000, 16, {2: 1.0}, quant_format="nvfp4"),
        ]
        plan = planner.plan(experts, TRACE_SHA)
        self.assertIn(plan.expert_to_gpu["0:0"][0], (0, 1))
        self.assertIn(plan.expert_to_gpu["0:1"][0], (2, 3))
        self.assertEqual(plan.predicted_cross_numa_bytes, 0)
        self.assertEqual(plan.quant_formats, ("nvfp4",))

    def test_quantized_size_and_headroom_gate_capacity(self):
        capacities = {gpu: 1200 for gpu in range(4)}
        headroom = {gpu: 300 for gpu in range(4)}
        planner = OfflineExpertPlacement(topology(), capacities, headroom)
        too_large = [QuantizedExpertProfile(0, 0, 1, 1, 1000, 16)]
        with self.assertRaises(PlacementError):
            planner.plan(too_large, TRACE_SHA)
        fits = [QuantizedExpertProfile(0, 0, 1, 1, 800, 16, quant_format="nvfp4")]
        self.assertEqual(planner.plan(fits, TRACE_SHA).gpu_memory_bytes[0], 800)

    def test_hot_expert_gets_redundant_copy(self):
        config = OfflineEPLBConfig(max_redundant_experts=1, replica_load_threshold=1.0)
        planner = OfflineExpertPlacement(topology(), {gpu: 10_000 for gpu in range(4)}, config=config)
        experts = [
            QuantizedExpertProfile(0, 0, 1000, 1, 500, 16, quant_format="w4a16"),
            QuantizedExpertProfile(0, 1, 10, 1, 500, 16, quant_format="w4a16"),
        ]
        plan = planner.plan(experts, TRACE_SHA)
        self.assertEqual(plan.replicas["0:0"], 1)
        self.assertEqual(plan.migration_bytes, 1500)


class OnlineControllerTests(unittest.TestCase):
    def test_runbook_thresholds_trigger_then_rollback(self):
        controller = OnlineEPLBController()
        decision = None
        for _ in range(3):
            decision = controller.observe(
                [100, 0, 0, 0], 1000, 100.0,
                candidate_benefit_fraction=0.10,
                candidate_benefit_us=400,
                migration_cost_us=100,
            )
        self.assertEqual(decision.action, "rebalance")
        for _ in range(2):
            self.assertEqual(controller.observe([1, 1, 1, 1], 1000, 106.0).action, "hold")
        self.assertEqual(controller.observe([1, 1, 1, 1], 1000, 106.0).action, "rollback")

    def test_low_sample_window_never_triggers(self):
        controller = OnlineEPLBController()
        for _ in range(5):
            decision = controller.observe(
                [100, 0, 0, 0], 999, 100.0,
                candidate_benefit_fraction=1.0,
                candidate_benefit_us=1000,
                migration_cost_us=1,
            )
        self.assertEqual(decision.action, "hold")
        self.assertIn("insufficient", decision.reason)

    def test_elapsed_window_is_eligible_and_cooldown_is_ten(self):
        controller = OnlineEPLBController()
        self.assertEqual(controller.config.cooldown_windows, 10)
        for _ in range(3):
            decision = controller.observe_cv(
                cv=0.5,
                requests=32,
                elapsed_ms=500,
                current_p99_ms=100.0,
                candidate_benefit_fraction=0.10,
                candidate_benefit_us=400,
                migration_cost_us=100,
            )
        self.assertEqual(decision.action, "rebalance")
        self.assertEqual(decision.cooldown_remaining, 10)
        for _ in range(10):
            decision = controller.observe_cv(
                cv=0.5,
                requests=32,
                elapsed_ms=500,
                current_p99_ms=100.0,
                candidate_benefit_fraction=0.10,
                candidate_benefit_us=400,
                migration_cost_us=100,
            )
            self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.cooldown_remaining, 0)

    def test_rollback_reference_uses_frozen_pre_migration_p99_ema(self):
        controller = OnlineEPLBController()
        for p99 in (1535.0, 1334.0, 1414.0):
            decision = controller.observe_cv(
                cv=0.5,
                requests=32,
                elapsed_ms=500,
                current_p99_ms=p99,
                candidate_benefit_fraction=0.10,
                candidate_benefit_us=400,
                migration_cost_us=100,
            )
        self.assertEqual(decision.action, "rebalance")
        reference = decision.rollback_reference_p99_ms
        self.assertIsNotNone(reference)
        self.assertGreater(reference, 1450.0)
        for _ in range(10):
            decision = controller.observe_cv(
                cv=0.5, requests=32, elapsed_ms=500, current_p99_ms=1522.0
            )
            self.assertEqual(decision.action, "hold")
            self.assertEqual(decision.rollback_reference_p99_ms, reference)


if __name__ == "__main__":
    unittest.main()
