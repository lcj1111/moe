import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rate_freeze", ROOT / "scripts" / "freeze_phase8_formal_rates.py")
RATE_FREEZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RATE_FREEZE)


class Phase8RateFreezeTests(unittest.TestCase):
    def test_matrix_has_common_conservative_open_loop_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_path = root / "audit.json"
            audit = {
                "status": "accepted",
                "rate_basis": "minimum observed",
                "cells": [{
                    "cell_id": "w1_c8", "input_tokens": 256,
                    "output_tokens": 128, "concurrency": 8, "requests": 32,
                    "minimum_observed_request_rps": 10.1234569,
                }],
            }
            RATE_FREEZE.write_json(audit_path, audit)
            matrix = RATE_FREEZE.build_formal_matrix(audit, 0.70, 42, audit_path)
            self.assertEqual(9, len(matrix["cells"]))
            open_loop = [row for row in matrix["cells"]
                         if row["arrival_mode"] != "closed_loop"]
            self.assertEqual({7.086419},
                             {row["request_rate_rps"] for row in open_loop})
            self.assertTrue(all("request_rate_rps" not in row for row in matrix["cells"]
                                if row["arrival_mode"] == "closed_loop"))
            self.assertEqual(9, len({row["stream_seed"] for row in matrix["cells"]}))

    def test_capacity_audit_rejects_nonzero_cache_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"cells": [{
                "id": "bad", "input_tokens": 256, "output_tokens": 16,
                "concurrency": 1, "requests": 2,
                "prefix_cache_pct": 50, "arrival_mode": "closed_loop",
            }]}), encoding="utf-8")
            (root / "schedule.json").write_text(json.dumps({
                "workload_matrix": str(matrix),
                "schedule": [{"candidate_id": "a", "repeat": 1}],
            }), encoding="utf-8")
            audit = RATE_FREEZE.audit_capacity(root, expected_repeats=1)
            self.assertEqual("rejected", audit["status"])
            self.assertTrue(any("prefix_cache_pct" in error for error in audit["errors"]))

    def test_safety_utilization_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            RATE_FREEZE.write_json(path, {"status": "accepted"})
            with self.assertRaises(ValueError):
                RATE_FREEZE.build_formal_matrix(
                    {"status": "accepted", "cells": []}, 1.0, 42, path)


if __name__ == "__main__":
    unittest.main()
