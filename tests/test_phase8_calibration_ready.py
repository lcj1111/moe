import unittest

from scripts.audit_phase8_calibration_ready import audit
from selector.kernel_db import KernelDatabase, KernelMeasurement
from selector.strategy_selector import StrategyCandidate


def candidate(candidate_id="fp8", precision="fp8", backend="triton",
              cost_backend=None):
    return StrategyCandidate(
        candidate_id=candidate_id,
        quant_format=precision,
        checkpoint="checkpoint",
        tp=1,
        dp=1,
        ep=1,
        gpu_mapping="mapping",
        eplb_policy="disabled",
        redundant_experts=0,
        kernel_backend=backend,
        cost_kernel_backend=cost_backend,
    )


def observation():
    return {
        "real_M_hist": {"1": 0.5, "8": 0.5},
        "measured_p99_ms_by_candidate": {"fp8": 10.0},
        "placement": {
            "measured_p99_bootstrap_95ci_by_candidate": {
                "fp8": {"low_ms": 9.0, "high_ms": 11.0, "n": 5},
            },
        },
    }


class CalibrationReadinessTests(unittest.TestCase):
    def test_ready_requires_all_buckets_and_repeated_evidence(self):
        db = KernelDatabase([
            KernelMeasurement("triton", {}, 1, precision="fp8", p95_us=1.0,
                              measured=True),
            KernelMeasurement("triton", {}, 8, precision="fp8", p95_us=2.0,
                              measured=True),
        ])
        result = audit([candidate()], [observation()], db)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(all(result["gate"].values()))

    def test_missing_bucket_is_a_blocker_not_a_synthetic_proxy(self):
        db = KernelDatabase([
            KernelMeasurement("triton", {}, 1, precision="fp8", p95_us=1.0,
                              measured=True),
        ])
        result = audit([candidate()], [observation()], db)
        self.assertEqual(result["status"], "blocked_missing_calibration_inputs")
        self.assertEqual(result["blockers"][0]["missing_kernel_m_buckets"], [8])
        self.assertFalse(result["gate"]["all_candidates_have_kernel_coverage"])

    def test_cost_backend_is_distinct_from_runtime_backend(self):
        db = KernelDatabase([
            KernelMeasurement("wna16_triton", {}, 1, precision="w4a16",
                              p95_us=1.0, measured=True),
            KernelMeasurement("wna16_triton", {}, 8, precision="w4a16",
                              p95_us=2.0, measured=True),
        ])
        w4_observation = observation()
        w4_observation["measured_p99_ms_by_candidate"] = {"w4": 10.0}
        w4_observation["placement"][
            "measured_p99_bootstrap_95ci_by_candidate"
        ] = {"w4": {"low_ms": 9.0, "high_ms": 11.0, "n": 5}}
        result = audit([
            candidate("w4", "w4a16", "triton", "wna16_triton")
        ], [w4_observation], db)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["candidate_coverage"][0]["runtime_kernel_backend"], "triton"
        )
        self.assertEqual(
            result["candidate_coverage"][0]["cost_kernel_backend"],
            "wna16_triton",
        )


if __name__ == "__main__":
    unittest.main()
