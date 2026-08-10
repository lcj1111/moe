import unittest

from scripts.audit_phase8_calibration_ready import audit
from selector.kernel_db import KernelDatabase, KernelMeasurement
from selector.strategy_selector import StrategyCandidate


def candidate(candidate_id="fp8", precision="fp8", backend="triton"):
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


if __name__ == "__main__":
    unittest.main()
