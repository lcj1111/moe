import importlib.util
import pathlib
import unittest


SCRIPT = (
    pathlib.Path(__file__).parents[1]
    / "scripts"
    / "build_fullset_truncation_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("build_fullset_truncation_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def manifest_row(row_id: str, *, answer: str = "A") -> dict:
    return {
        "id": row_id,
        "benchmark": "mmlu_pro",
        "score_type": "multiple_choice",
        "answer": answer,
        "max_tokens": 4000,
        "protocol": "official_cot_fewshot_full",
    }


def result_row(row_id: str, *, truncated: bool = False) -> dict:
    return {
        "id": row_id,
        "benchmark": "mmlu_pro",
        "score_type": "multiple_choice",
        "expected": "A",
        "max_tokens": 4000,
        "correct": None if truncated else True,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
        "error": None,
    }


class ValidateBaseResultsTest(unittest.TestCase):
    def test_accepts_complete_ordered_results(self) -> None:
        MODULE.validate_base_results(
            [manifest_row("a"), manifest_row("b")],
            [result_row("a"), result_row("b", truncated=True)],
        )

    def test_rejects_id_order_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "ids/order"):
            MODULE.validate_base_results(
                [manifest_row("a"), manifest_row("b")],
                [result_row("b"), result_row("a")],
            )

    def test_rejects_identity_drift(self) -> None:
        changed = result_row("a")
        changed["expected"] = "B"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            MODULE.validate_base_results([manifest_row("a")], [changed])

    def test_rejects_scored_truncation(self) -> None:
        changed = result_row("a", truncated=True)
        changed["correct"] = False
        with self.assertRaisesRegex(ValueError, "correct=null"):
            MODULE.validate_base_results([manifest_row("a")], [changed])

    def test_rejects_request_failure(self) -> None:
        changed = result_row("a")
        changed["error"] = "HTTP 500"
        with self.assertRaisesRegex(ValueError, "request failure"):
            MODULE.validate_base_results([manifest_row("a")], [changed])


if __name__ == "__main__":
    unittest.main()
