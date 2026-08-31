# 作用：验证 full-set 双分片脚本的参数与安全约束。
import pathlib
import unittest


class FullsetQualityPairScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            pathlib.Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_fullset_quality_pair.sh"
        ).read_text(encoding="utf-8")

    def test_usage_lists_all_supported_formats(self) -> None:
        self.assertIn("<bf16|nvfp4|fp8>", self.script)

    def test_bf16_uses_two_tp4_services_and_eager_mode(self) -> None:
        bf16_block = self.script.split("  bf16)", 1)[1].split("  nvfp4)", 1)[0]
        self.assertIn("/data/models/REAP/models/Qwen3.6-35B-A3B", bf16_block)
        self.assertIn("TP_SIZE=4", bf16_block)
        self.assertIn('GPU_A="0,1,2,3"', bf16_block)
        self.assertIn('GPU_B="4,5,6,7"', bf16_block)
        self.assertIn("EXTRA_ARGS=(--enforce-eager)", bf16_block)


if __name__ == "__main__":
    unittest.main()
