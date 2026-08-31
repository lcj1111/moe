# 作用：验证服务请求客户端的输入、统计和错误处理。
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("service_workload", ROOT / "clients" / "smoke.py")
CLIENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CLIENT)


class ServiceWorkloadClientTests(unittest.TestCase):
    def test_poisson_schedule_is_seeded_and_monotonic(self):
        first = CLIENT.arrival_offsets("poisson", 16, 4, 8.0, 42)
        second = CLIENT.arrival_offsets("poisson", 16, 4, 8.0, 42)
        self.assertEqual(first, second)
        self.assertEqual(0.0, first[0])
        self.assertTrue(all(right > left for left, right in zip(first, first[1:])))

    def test_burst_schedule_preserves_average_offered_rate(self):
        offsets = CLIENT.arrival_offsets("burst", 8, 4, 2.0, 42)
        self.assertEqual([0.0] * 4 + [2.0] * 4, offsets)

    def test_equal_arrival_offsets_form_dispatch_batches(self):
        offsets = [0.0, 0.0, 0.0, 2.0, 2.0, 4.0]
        self.assertEqual([
            (0.0, [0, 1, 2]),
            (2.0, [3, 4]),
            (4.0, [5]),
        ], CLIENT.arrival_batches(offsets))

    def test_poisson_arrivals_remain_single_request_batches(self):
        offsets = CLIENT.arrival_offsets("poisson", 8, 4, 2.0, 42)
        batches = CLIENT.arrival_batches(offsets)
        self.assertEqual(8, len(batches))
        self.assertTrue(all(len(request_ids) == 1 for _, request_ids in batches))

    def test_execute_burst_uses_one_shared_batch_timestamp(self):
        args = SimpleNamespace(
            base_url="http://unused", model="unused", input_tokens=16,
            output_tokens=4, seed=42, timeout=1, concurrency=4,
            arrival_mode="burst", requests=4, request_rate=2.0,
            stream_seed=42,
        )
        prompts = [{"text": "x", "input_tokens_actual": 16,
                    "cache_salt": "s", "prompt_sha256": "h"}
                   for _ in range(4)]

        def fake_request(*values):
            return {"request_id": values[4], "submitted_offset_s": values[13]}

        with mock.patch.object(CLIENT, "one_request", side_effect=fake_request):
            records = CLIENT.execute_requests(args, prompts)
        self.assertEqual(1, len({row["submitted_offset_s"] for row in records}))

    def test_open_loop_requires_positive_rate(self):
        with self.assertRaises(ValueError):
            CLIENT.arrival_offsets("poisson", 4, 2, None, 42)
        with self.assertRaises(ValueError):
            CLIENT.arrival_offsets("burst", 4, 2, 0.0, 42)

    def test_cached_tokens_are_read_from_openai_usage_details(self):
        usage = {"prompt_tokens": 256,
                 "prompt_tokens_details": {"cached_tokens": 128}}
        self.assertEqual(128, CLIENT.cached_tokens_from_usage(usage))
        self.assertIsNone(CLIENT.cached_tokens_from_usage({"prompt_tokens": 256}))

    def test_realizable_cache_ratio_respects_hybrid_page_granularity(self):
        self.assertEqual(0, CLIENT.realizable_cached_tokens(256, 256, 100, 1056))
        self.assertEqual(2112, CLIENT.realizable_cached_tokens(4224, 2112, 50, 1056))
        self.assertEqual(3168, CLIENT.realizable_cached_tokens(4224, 4224, 100, 1056))
        self.assertEqual(0, CLIENT.realizable_cached_tokens(4224, 4224, 0, 1056))

    def test_common_prefix_and_peak_in_flight(self):
        self.assertEqual(2, CLIENT.common_prefix_tokens([[1, 2, 3], [1, 2, 4]]))
        rows = [
            {"started_offset_s": 0.0, "finished_offset_s": 2.0},
            {"started_offset_s": 1.0, "finished_offset_s": 3.0},
            {"started_offset_s": 3.1, "finished_offset_s": 4.0},
        ]
        self.assertEqual(2, CLIENT.peak_in_flight(rows))

    def test_open_loop_timing_separates_dispatch_from_worker_queue(self):
        rows = [{
            "scheduled_offset_s": 1.0,
            "submitted_offset_s": 1.01,
            "started_offset_s": 1.50,
            "finished_offset_s": 2.50,
            "e2e_ms": 1000.0,
            "ttft_ms": 100.0,
        }]
        timing = CLIENT.arrival_timing(rows, "poisson")
        self.assertAlmostEqual(0.01, timing["dispatch_lag_s"][0])
        self.assertAlmostEqual(0.49, timing["queue_delay_s"][0])
        self.assertAlmostEqual(0.50, timing["service_start_lag_s"][0])
        self.assertAlmostEqual(1500.0, timing["offered_e2e_ms"][0])
        self.assertAlmostEqual(600.0, timing["offered_ttft_ms"][0])

    def test_closed_loop_offered_latency_equals_service_latency(self):
        rows = [{
            "started_offset_s": 0.0, "finished_offset_s": 1.0,
            "e2e_ms": 1000.0, "ttft_ms": 100.0,
        }]
        timing = CLIENT.arrival_timing(rows, "closed_loop")
        self.assertEqual([1000.0], timing["offered_e2e_ms"])
        self.assertEqual([100.0], timing["offered_ttft_ms"])


if __name__ == "__main__":
    unittest.main()
