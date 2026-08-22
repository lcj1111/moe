import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "compare_fullset_quality_results.py"
SPEC = importlib.util.spec_from_file_location("compare_fullset_quality_results", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(row_id, correct, benchmark="mmlu_pro"):
    return {
        "id": row_id,
        "benchmark": benchmark,
        "expected": "A",
        "score_type": "choice",
        "correct": correct,
    }


class CompareFullsetQualityResultsTest(unittest.TestCase):
    def test_uses_common_scored_denominator(self):
        left = [row("a", True), row("b", None), row("c", False, "ceval")]
        right = [row("a", False), row("b", True), row("c", True, "ceval")]
        result = MODULE.compare(left, right, "nvfp4", "fp8")
        overall = result["common_denominator"]["overall"]
        self.assertEqual(overall["common_scored"], 2)
        self.assertEqual(overall["nvfp4"]["correct"], 1)
        self.assertEqual(overall["fp8"]["correct"], 1)
        self.assertEqual(result["unfinished"]["union"], 1)
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_mismatched_ids(self):
        with self.assertRaisesRegex(ValueError, "ID 集合"):
            MODULE.compare([row("a", True)], [row("b", True)], "left", "right")

    def test_rejects_identity_drift(self):
        changed = row("a", True)
        changed["expected"] = "B"
        with self.assertRaisesRegex(ValueError, "expected"):
            MODULE.compare([row("a", True)], [changed], "left", "right")

    def test_three_format_common_denominator(self):
        datasets = {
            "bf16": [row("a", True), row("b", None), row("c", False, "ceval")],
            "fp8": [row("a", False), row("b", True), row("c", True, "ceval")],
            "nvfp4": [row("a", True), row("b", False), row("c", None, "ceval")],
        }
        result = MODULE.compare_many(datasets)
        overall = result["common_denominator"]["overall"]
        self.assertEqual(overall["common_scored"], 1)
        self.assertEqual(overall["formats"]["bf16"]["correct"], 1)
        self.assertEqual(overall["formats"]["fp8"]["correct"], 0)
        self.assertEqual(overall["formats"]["nvfp4"]["correct"], 1)
        self.assertEqual(result["unfinished"]["union_all_formats"], 2)
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
