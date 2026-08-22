import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "merge_fullset_quality_results.py"
SPEC = importlib.util.spec_from_file_location("merge_fullset_quality_results", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(row_id, *, truncated=False, correct=True, benchmark="mmlu_pro"):
    return {
        "id": row_id,
        "benchmark": benchmark,
        "expected": "A",
        "prediction": None if truncated else "A",
        "correct": None if truncated else correct,
        "score_type": "choice",
        "truncated": truncated,
        "error": None,
    }


class MergeFullsetQualityResultsTest(unittest.TestCase):
    def test_replaces_only_truncated_rows_and_preserves_manifest_order(self):
        manifest = [{"id": "b"}, {"id": "a"}, {"id": "c"}]
        base = [row("a"), row("b", truncated=True), row("c", truncated=True)]
        rerun = [row("b"), row("c", truncated=True)]

        merged, audit = MODULE.merge_results(manifest, base, rerun)

        self.assertEqual([item["id"] for item in merged], ["b", "a", "c"])
        self.assertEqual(
            [item["merge_source"] for item in merged], ["rerun", "base", "rerun"]
        )
        self.assertEqual(audit["replaced_records"], 2)
        self.assertEqual(audit["truncated"], 1)
        self.assertEqual(audit["scored"], 2)
        self.assertTrue(all(audit["checks"].values()))

    def test_rejects_rerun_of_non_truncated_base_row(self):
        manifest = [{"id": "a"}]
        with self.assertRaisesRegex(ValueError, "续跑 id"):
            MODULE.merge_results(manifest, [row("a")], [row("a")])

    def test_rejects_missing_rerun_for_base_truncation(self):
        manifest = [{"id": "a"}]
        with self.assertRaisesRegex(ValueError, "续跑 id"):
            MODULE.merge_results(manifest, [row("a", truncated=True)], [])

    def test_rejects_identity_mismatch(self):
        manifest = [{"id": "a"}]
        changed = row("a")
        changed["expected"] = "B"
        with self.assertRaisesRegex(ValueError, "expected"):
            MODULE.merge_results(manifest, [row("a", truncated=True)], [changed])


if __name__ == "__main__":
    unittest.main()
