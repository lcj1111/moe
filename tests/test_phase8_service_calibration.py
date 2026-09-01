# 作用：验证服务修正模型的非负拟合与留族划分。
import unittest

from scripts.calibrate_phase8_service_model import (
    fit_nonnegative_affine,
    workload_family,
)
from selector.strategy_selector import WorkloadObservation


class Phase8ServiceCalibrationTests(unittest.TestCase):
    def test_weighted_affine_recovers_exact_transparent_mapping(self):
        fit = fit_nonnegative_affine([
            (1.0, 5.0, 1.0),
            (2.0, 7.0, 2.0),
            (4.0, 11.0, 0.5),
        ])
        self.assertAlmostEqual(fit["scale"], 2.0)
        self.assertAlmostEqual(fit["intercept_ms"], 3.0)
        self.assertAlmostEqual(fit["weighted_train_rmse_ms"], 0.0)

    def test_workload_family_is_explicit_grouping_key(self):
        observation = WorkloadObservation(
            {1: 1.0}, placement={"workload_id": "w3_c16"}
        )
        self.assertEqual(workload_family(observation), "w3")

    def test_negative_slope_is_clamped_for_monotonicity(self):
        fit = fit_nonnegative_affine([
            (1.0, 10.0, 1.0),
            (2.0, 5.0, 1.0),
        ])
        self.assertEqual(fit["scale"], 0.0)
        self.assertEqual(fit["intercept_ms"], 7.5)


if __name__ == "__main__":
    unittest.main()
