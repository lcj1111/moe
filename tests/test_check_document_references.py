# 作用：验证文档链接与仓库路径检查规则。
from pathlib import Path
import importlib.util
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_document_references.py"
SPEC = importlib.util.spec_from_file_location("check_document_references", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DocumentReferenceCheckTest(unittest.TestCase):
    def test_valid_links_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "docs").mkdir()
            (root / "scripts").mkdir()
            (root / "scripts" / "run.py").write_text("", encoding="utf-8")
            (root / "docs" / "detail.md").write_text("# 详情\n", encoding="utf-8")
            (root / "configs" / "README.md").write_text(
                "[返回入口](../README.md)\n", encoding="utf-8"
            )
            (root / "README.md").write_text(
                "[详情](docs/detail.md) 与 `scripts/run.py`\n", encoding="utf-8"
            )
            self.assertEqual(MODULE.check(root), [])

    def test_missing_link_and_repo_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "README.md").write_text(
                "[缺失](docs/missing.md) 与 `scripts/missing.py`\n",
                encoding="utf-8",
            )
            errors = MODULE.check(root)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("本地链接不存在" in error for error in errors))
            self.assertTrue(any("仓库路径不存在" in error for error in errors))

    def test_missing_local_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "![拓扑图](docs/missing.png)\n", encoding="utf-8"
            )
            errors = MODULE.check(root)
            self.assertEqual(len(errors), 1)
            self.assertIn("本地链接不存在", errors[0])

    def test_local_dependencies_are_not_project_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (".venv", "third_party", ".git"):
                (root / name).mkdir()
                (root / name / "README.md").write_text(
                    "[外部包内部链接](missing.md)", encoding="utf-8"
                )
            self.assertEqual(MODULE.check(root), [])


if __name__ == "__main__":
    unittest.main()
