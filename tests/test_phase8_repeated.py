import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
