# 作用：验证路由稳定性诊断指标和 Gate。
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase8_route_stability_diagnostic",
    ROOT / "scripts" / "analyze_phase8_route_stability_diagnostic.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RouteStabilityDiagnosticTest(unittest.TestCase):
    def test_distribution_distance_uses_normalized_counts(self) -> None:
        left = [[[10.0, 20.0, 30.0]]]
        right = [[[20.0, 40.0, 60.0]]]
        result = MODULE.distribution_distance(left, right)
        self.assertEqual(result["layers"], 1)
        self.assertAlmostEqual(result["tv_p95"], 0.0)

    def _case(
        self,
        short_candidate: list[float],
        long_candidate: list[float],
        legacy_tv: float = 0.0,
        confirm_sampling_scope: bool = False,
    ) -> dict:
        identity = [10.0, 0.0, 0.0, 0.0]
        phases = {
            "短输出同输入对照": ["sa1", "sb1", "sa2", "sb2"],
            "长输出复现对照": ["la1", "lb1", "la2", "lb2"],
        }
        plan = {
            "诊断负载": {
                name: {"测量阶段": values} for name, values in phases.items()
            },
            "分类阈值": {
                "legacy_vs_record_time_tv_p95_max": 0.000001,
                "短输出placement_excess_tv_p95_max": 0.005,
                "长输出placement_excess_tv_p95_max": 0.005,
            },
        }
        if confirm_sampling_scope:
            plan["确认归因"] = {
                "enabled": True,
                "稳定时归因": "统计采样与阶段顺序口径",
            }
        distributions = {
            "sa1": identity,
            "sa2": identity,
            "sb1": short_candidate,
            "sb2": short_candidate,
            "la1": identity,
            "la2": identity,
            "lb1": long_candidate,
            "lb2": long_candidate,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            ranges = []
            for index, (arm, names) in enumerate(phases.items()):
                del index
                for name in names:
                    records.append(
                        {
                            "legacy_vs_record_time_tv_by_model": [[legacy_tv]],
                            "record_time_mismatched_slots": [0],
                            "current_map_sha256": ["map"],
                            "record_time_logical_models": [[distributions[name]]],
                        }
                    )
                    ranges.append(
                        {
                            "phase": name,
                            "arm": arm,
                            "role": "measured",
                            "start_line": len(records) - 1,
                            "end_line": len(records),
                        }
                    )
                    phase_dir = root / name
                    phase_dir.mkdir()
                    (phase_dir / "requests.jsonl").write_text(
                        json.dumps(
                            {
                                "request_id": 0,
                                "prompt_sha256": "prompt",
                                "response_sha256": name,
                                "cached_tokens": 128,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            (root / "route_diagnostic_windows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )
            (root / "phase_ranges.json").write_text(
                json.dumps({"phases": ranges}), encoding="utf-8"
            )
            return MODULE.analyze(plan, root)

    def test_classifies_short_input_actual_route_change(self) -> None:
        result = self._case(
            short_candidate=[0.0, 10.0, 0.0, 0.0],
            long_candidate=[0.0, 10.0, 0.0, 0.0],
        )
        self.assertTrue(
            result["classification"]["actual_route_change_under_short_matched_input"]
        )

    def test_classifies_long_generation_divergence(self) -> None:
        result = self._case(
            short_candidate=[10.0, 0.0, 0.0, 0.0],
            long_candidate=[0.0, 10.0, 0.0, 0.0],
        )
        self.assertFalse(
            result["classification"]["actual_route_change_under_short_matched_input"]
        )
        self.assertTrue(
            result["classification"]["long_generation_continuation_divergence"]
        )

    def test_classifies_steady_reconstruction_error(self) -> None:
        result = self._case(
            short_candidate=[10.0, 0.0, 0.0, 0.0],
            long_candidate=[10.0, 0.0, 0.0, 0.0],
            legacy_tv=0.01,
        )
        self.assertTrue(
            result["classification"]["statistical_scope_or_reconstruction_error"]
        )

    def test_confirmation_attributes_stable_result_to_sampling_scope(self) -> None:
        identity = [10.0, 0.0, 0.0, 0.0]
        result = self._case(
            short_candidate=identity,
            long_candidate=identity,
            confirm_sampling_scope=True,
        )
        self.assertTrue(result["classification"]["sampling_or_phase_order_scope"])
        self.assertTrue(
            result["classification"]["statistical_scope_or_reconstruction_error"]
        )


if __name__ == "__main__":
    unittest.main()
