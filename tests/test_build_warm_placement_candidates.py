# 作用：验证暖态 placement 候选的迁移约束与映射一致性。
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_warm_placement_candidates",
    ROOT / "scripts" / "build_warm_placement_candidates.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class WarmPlacementCandidatesTest(unittest.TestCase):
    def test_limited_swaps_preserve_permutation_and_improve_cv(self):
        counts = [[100.0, 90.0, 80.0, 70.0, 4.0, 3.0, 2.0, 1.0]]
        identity = MODULE.identity_map(1, 8)
        candidate = MODULE.limited_swap_map(counts, ranks=2, moved_slots_per_layer=2)
        MODULE.validate_map(candidate, layers=1, experts=8)
        before = MODULE.map_metrics(identity, counts, ranks=2)
        after = MODULE.map_metrics(candidate, counts, ranks=2)
        self.assertLess(after["layer_rank_cv_median"], before["layer_rank_cv_median"])
        self.assertEqual(MODULE.moved_slots(candidate), 2)

    def test_full_lpt_keeps_equal_capacity(self):
        counts = [[float(index + 1) for index in range(16)]]
        candidate = MODULE.full_lpt_map(counts, ranks=4)
        MODULE.validate_map(candidate, layers=1, experts=16)
        self.assertEqual(len(candidate[0]), 16)
        self.assertEqual(sorted(candidate[0]), list(range(16)))


if __name__ == "__main__":
    unittest.main()
