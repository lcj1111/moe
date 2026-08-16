import importlib.util
import unittest
from pathlib import Path

from clients import smoke


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AGGREGATOR = load_script("aggregate_phase8_repeated.py")
INDEPENDENT = load_script("evaluate_phase8_independent_selector.py")
ATTACH = load_script("attach_phase8_predecision_state.py")


def record(request_id: int, started: float, finished: float,
           submitted: float) -> dict:
    return {
        "request_id": request_id,
        "started_offset_s": started,
        "finished_offset_s": finished,
        "submitted_offset_s": submitted,
        "scheduled_offset_s": submitted,
        "e2e_ms": (finished - started) * 1000,
        "ttft_ms": 10.0,
        "cached_tokens": 50,
        "prompt_tokens_reported": 100,
    }


def measurement(a: float, b: float) -> dict:
    def candidate(value: float) -> dict:
        return {"metrics": {"e2e_p99_ms": {
            "median": value,
            "bootstrap_95ci_low": value,
            "bootstrap_95ci_high": value,
        }}}
    return {"a": candidate(a), "b": candidate(b)}


def row(workload_id: str, input_tokens: int, a: float, b: float,
        phase: str = "pre_decision") -> dict:
    return {
        "workload_id": workload_id,
        "input_tokens": input_tokens,
        "output_tokens": 128,
        "concurrency": 4,
        "prefix_cache_pct": 50,
        "arrival_mode": "poisson",
        "request_rate_rps": 2.0,
        "selector_state": {
            "observation_phase": phase,
            "decision_eligible": phase == "pre_decision",
            "server_queue_depth_available": True,
        },
        "measurements": measurement(a, b),
    }


class SelectorStateTests(unittest.TestCase):
    def test_prometheus_parser_sums_labeled_series(self):
        parsed = smoke.parse_prometheus_sample("""
        # HELP vllm:num_requests_waiting waiting
        vllm:num_requests_waiting{model_name="a"} 2
        vllm:num_requests_waiting{model_name="b"} 3
        vllm:num_requests_running 4
        broken text
        """)
        summary = smoke.summarize_service_metrics(
            [{"metrics": parsed}], "http://host/metrics", True)
        self.assertEqual(5.0, parsed["vllm:num_requests_waiting"])
        self.assertEqual(5.0, summary["num_requests_waiting"]["p95"])
        self.assertTrue(summary["server_queue_depth_available"])

    def test_selector_state_marks_only_pre_decision_as_eligible(self):
        records = [record(0, 0.1, 0.3, 0.0), record(1, 0.2, 0.5, 0.1)]
        telemetry = {
            "server_queue_depth_available": True,
            "num_requests_waiting": {"p95": 2.0},
            "num_requests_running": {"p95": 3.0},
            "kv_cache_usage_perc": {"p95": 0.4},
        }
        state = smoke.build_selector_state(
            records, "poisson", 2, "pre_decision", telemetry)
        self.assertTrue(state["decision_eligible"])
        self.assertEqual(0.5, state["actual_cache_hit_ratio"])
        self.assertEqual(100.0, state["client_queue_delay_ms"]["p95"])
        state = smoke.build_selector_state(
            records, "poisson", 2, "post_workload", telemetry)
        self.assertFalse(state["decision_eligible"])

    def test_aggregator_preserves_queue_and_selector_fields(self):
        summary = {
            "prefix_cache": {"actual_cached_token_ratio": 0.5},
            "arrival": {
                "queue_delay_s": {"p95": 0.2},
                "peak_in_flight": 7,
                "service_start_rate_realized_rps": 3.5,
            },
            "service_telemetry": {
                "num_requests_waiting": {"p95": 2.0},
                "coverage_ratio": 1.0,
            },
            "selector_state": {"observation_phase": "post_workload"},
        }
        value = AGGREGATOR.repeat_telemetry(summary)
        self.assertEqual(0.2, value["client_queue_delay_p95_s"])
        self.assertEqual(2.0, value["server_queue_waiting_p95"])
        self.assertEqual("post_workload", value["selector_state"]["observation_phase"])

    def test_independent_gate_accepts_disjoint_pre_decision_inputs(self):
        training = {"status": "accepted", "rows": [row("train", 256, 10, 20)]}
        test = {"status": "accepted", "rows": [row("test", 512, 11, 22)]}
        config = {
            "status": "frozen",
            "strict_match": ["arrival_mode"],
            "require_server_queue_telemetry": True,
            "numeric_features": [
                {"path": "input_tokens", "transform": "log2", "weight": 1.0}
            ],
            "gates": {
                "median_regret_pct_max": 5.0,
                "p95_regret_pct_max": 10.0,
                "decision_overhead_pct_max": 100.0,
                "infeasible_misselection_rate_max": 0.0,
            },
        }
        result = INDEPENDENT.evaluate(training, test, config, 1)
        self.assertEqual("accepted", result["status"])
        self.assertEqual(0.0, result["metrics"]["p95_regret_pct"])

    def test_independent_gate_rejects_post_workload_state(self):
        training = {"status": "accepted", "rows": [row(
            "train", 256, 10, 20, phase="post_workload")]}
        test = {"status": "accepted", "rows": [row("test", 512, 11, 22)]}
        config = {
            "status": "frozen", "strict_match": ["arrival_mode"],
            "require_server_queue_telemetry": True,
            "numeric_features": [{"path": "input_tokens", "transform": "log2"}],
            "gates": {},
        }
        with self.assertRaisesRegex(ValueError, "不是决策前窗口"):
            INDEPENDENT.evaluate(training, test, config, 1)

    def test_distance_allows_only_symmetric_not_applicable_feature(self):
        left = {"workload_id": "left", "state": {"queue": None}}
        right = {"workload_id": "right", "state": {"queue": None}}
        feature = [{
            "path": "state.queue", "transform": "log1p",
            "allow_both_null": True,
        }]
        self.assertEqual(0.0, INDEPENDENT.distance(left, right, feature))
        right["state"]["queue"] = 0.0
        with self.assertRaisesRegex(ValueError, "只有一侧缺失"):
            INDEPENDENT.distance(left, right, feature)

    def test_attach_keeps_outcomes_and_adds_incumbent_state(self):
        outcome_row = row("train", 256, 10, 20)
        outcome_row.pop("selector_state")
        aggregate = {"status": "accepted", "rows": [outcome_row]}
        states = {
            "status": "accepted",
            "incumbent_candidate_id": "a",
            "matrix_sha256": "matrix",
            "client_sha256": "client",
            "states": {"train": {
                "workload": {key: outcome_row.get(key) for key in (
                    "input_tokens", "output_tokens", "concurrency", "prefix_cache_pct",
                    "arrival_mode", "request_rate_rps")},
                "selector_state": {
                    "observation_phase": "pre_decision",
                    "decision_eligible": True,
                },
                "summary": "/summary.json",
                "summary_sha256": "summary",
            }},
        }
        result = ATTACH.attach(aggregate, states)
        self.assertEqual(measurement(10, 20), result["rows"][0]["measurements"])
        self.assertTrue(result["rows"][0]["selector_state"]["decision_eligible"])
        self.assertEqual("a", result["predecision_state_manifest"][
            "incumbent_candidate_id"])


if __name__ == "__main__":
    unittest.main()
