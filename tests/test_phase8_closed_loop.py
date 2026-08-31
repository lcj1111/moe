# 作用：验证 trigger、cooldown、提交和 rollback 闭环。
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase8_selector_closed_loop_acceptance.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("phase8_closed_loop", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ClosedLoopMetricTest(unittest.TestCase):
    def test_parses_layer_expert_and_rank_cv_separately(self) -> None:
        lines = "\n".join([
            "(Worker_TP0_EP0 pid=1) [QTOPOMOE_EPLB_LOAD_WINDOW] "
            "call=2 cv=0.40 expert_cv=0.40 rank_cv=0.08 generation=0 action=hold",
            "(Worker_TP0_EP0 pid=1) [QTOPOMOE_EPLB_LOAD_WINDOW] "
            "call=2 cv=0.60 expert_cv=0.60 rank_cv=0.10 generation=0 action=hold",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(lines, encoding="utf-8")
            metrics = MODULE.load_metrics_since(path, 0)
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["expert_load_cv_layer_median"], 0.5)
        self.assertAlmostEqual(metrics["rank_load_cv"], 0.09)


if __name__ == "__main__":
    unittest.main()
