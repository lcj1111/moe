import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("run_phase8_repeated.py")
AGGREGATOR = load_script("aggregate_phase8_repeated.py")


class RepeatedPlanTests(unittest.TestCase):
    def test_schedule_is_seeded_complete_and_unique(self):
        plan = {
            "seed": 42,
            "repeats_per_candidate": 5,
            "candidates": [{"candidate_id": name} for name in ("a", "b", "c", "d")],
        }
        first = RUNNER.build_schedule(plan)
        second = RUNNER.build_schedule(plan)
        self.assertEqual(first, second)
        self.assertEqual(20, len(first))
        self.assertEqual(20, len({(row["candidate_id"], row["repeat"])
                                  for row in first}))
        self.assertEqual(list(range(1, 21)), [row["order"] for row in first])

    def test_bootstrap_median_is_deterministic_and_bounded(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        first = AGGREGATOR.bootstrap_median(values, seed=42, samples=2000)
        second = AGGREGATOR.bootstrap_median(values, seed=42, samples=2000)
        self.assertEqual(first, second)
        self.assertEqual(30.0, first["median"])
        self.assertLessEqual(min(values), first["bootstrap_95ci_low"])
        self.assertGreaterEqual(max(values), first["bootstrap_95ci_high"])

    def test_service_command_enables_measured_prefix_cache_usage(self):
        plan = {
            "host": "127.0.0.1", "port": 31540, "max_model_len": 65536,
            "max_num_seqs": 8, "gpu_memory_utilization": 0.9,
        }
        candidate = {
            "model_path": "/model", "tp": 2, "dp": 1,
            "enable_expert_parallel": False, "moe_backend": "triton",
            "numa_args": [],
        }
        command = RUNNER.service_command(plan, candidate, "/venv/bin/vllm", "served")
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-prompt-tokens-details", command)

    def test_controlled_summary_gate_checks_cache_and_frozen_rate(self):
        cell = {
            "prefix_cache_pct": 50,
            "arrival_mode": "poisson",
            "request_rate_rps": 2.5,
        }
        summary = {
            "server_prompt_tokens_exact": True,
            "prefix_cache": {
                "target_pct": 50,
                "usage_details_complete": True,
                "ratio_gate": True,
            },
            "arrival": {
                "mode": "poisson",
                "schedule_gate": True,
                "request_rate_target_rps": 2.5,
            },
        }
        self.assertEqual([], AGGREGATOR.controlled_summary_errors(
            summary, cell, "candidate/cell"))
        summary["arrival"]["request_rate_target_rps"] = 3.0
        summary["prefix_cache"]["ratio_gate"] = False
        errors = AGGREGATOR.controlled_summary_errors(
            summary, cell, "candidate/cell")
        self.assertTrue(any("cache ratio" in error for error in errors))
        self.assertTrue(any("frozen request-rate" in error for error in errors))

    def test_uncontrolled_legacy_summary_is_not_reinterpreted(self):
        self.assertEqual([], AGGREGATOR.controlled_summary_errors(
            {}, {"id": "legacy"}, "candidate/legacy"))

    def test_controlled_latency_uses_scheduled_arrival_clock(self):
        summary = {
            "e2e_ms": {"p99": 100.0},
            "offered_e2e_ms": {"p99": 250.0},
            "ttft_ms": {"p99": 20.0},
            "offered_ttft_ms": {"p99": 170.0},
        }
        cell = {"prefix_cache_pct": 0, "arrival_mode": "poisson"}
        self.assertEqual(250.0, AGGREGATOR.latency_metric(
            summary, cell, "e2e_ms")["p99"])
        self.assertEqual(170.0, AGGREGATOR.latency_metric(
            summary, cell, "ttft_ms")["p99"])
        self.assertEqual(100.0, AGGREGATOR.latency_metric(
            summary, {"id": "legacy"}, "e2e_ms")["p99"])

    def test_schedule_failure_publishes_terminal_failed_status(self):
        schedule = [{"candidate_id": "a", "repeat": 1, "order": 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            with mock.patch.object(RUNNER, "run_one", side_effect=RuntimeError("gate")):
                with self.assertRaisesRegex(RuntimeError, "gate"):
                    RUNNER.execute_schedule(
                        {"cooldown_seconds": 0}, {"a": {}}, schedule, {}, ROOT,
                        output_root, "/vllm", "/python", {})
            status = json.loads((output_root / "status.json").read_text())
        self.assertEqual("failed", status["status"])
        self.assertEqual(schedule[0], status["current"])
        self.assertEqual(0, status["completed_runs"])
        self.assertIn("RuntimeError('gate')", status["failure"])


if __name__ == "__main__":
    unittest.main()
