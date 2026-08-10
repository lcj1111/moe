import importlib.util
import unittest
from pathlib import Path


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

    def test_common_prefix_and_peak_in_flight(self):
        self.assertEqual(2, CLIENT.common_prefix_tokens([[1, 2, 3], [1, 2, 4]]))
        rows = [
            {"started_offset_s": 0.0, "finished_offset_s": 2.0},
            {"started_offset_s": 1.0, "finished_offset_s": 3.0},
            {"started_offset_s": 3.1, "finished_offset_s": 4.0},
        ]
        self.assertEqual(2, CLIENT.peak_in_flight(rows))


if __name__ == "__main__":
    unittest.main()
