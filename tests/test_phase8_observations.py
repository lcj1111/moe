# 作用：验证 workload 观测聚合与置信区间计算。
import unittest

from scripts.build_phase8_observations import (
    _measurement_median_and_ci,
    build_observations,
)


class Phase8ObservationTests(unittest.TestCase):
    def test_reads_single_pass_scalar(self):
        median, ci = _measurement_median_and_ci({"e2e_p99_ms": 12.5})
        self.assertEqual(median, 12.5)
        self.assertIsNone(ci)

    def test_reads_repeated_median_and_interval(self):
        median, ci = _measurement_median_and_ci({"metrics": {"e2e_p99_ms": {
            "median": 10.0,
            "bootstrap_95ci_low": 8.0,
            "bootstrap_95ci_high": 12.0,
            "bootstrap_samples": 10000,
            "n": 5,
        }}})
        self.assertEqual(median, 10.0)
        self.assertEqual(ci, {
            "low_ms": 8.0,
            "high_ms": 12.0,
            "bootstrap_samples": 10000,
            "n": 5,
        })

    def test_builds_repeated_observation_without_oracle_lookup(self):
        buckets = {"records": [{
            "input_tokens": 256,
            "output_tokens": 128,
            "concurrency": 1,
            "prefix_cache_pct": 0,
            "arrival_mode": "closed_loop",
            "prefill_m_bucket": 256,
            "decode_m_bucket": 1,
            "real_M_hist": {"1": 0.25, "256": 0.75},
        }]}
        repeated = {
            "status": "accepted",
            "schema_version": "qtopomoe.phase8_repeated_bootstrap.v1",
            "rows": [{
                "workload_id": "w1_c1",
                "input_tokens": 256,
                "output_tokens": 128,
                "concurrency": 1,
                "measurements": {"candidate": {"metrics": {"e2e_p99_ms": {
                    "median": 20.0,
                    "bootstrap_95ci_low": 18.0,
                    "bootstrap_95ci_high": 22.0,
                    "bootstrap_samples": 10000,
                    "n": 5,
                }}}},
            }],
        }
        result = build_observations(
            buckets, repeated,
            {"communication_bytes": 1000.0, "route_hist": {"0": 1.0}},
            route_token_count=100,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["measured_p99_ms_by_candidate"], {"candidate": 20.0})
        self.assertEqual(result[0]["communication_bytes"], 3840.0)
        ci = result[0]["placement"]["measured_p99_bootstrap_95ci_by_candidate"]
        self.assertEqual(ci["candidate"]["n"], 5)
        self.assertNotIn("oracle", result[0]["placement"])


if __name__ == "__main__":
    unittest.main()
