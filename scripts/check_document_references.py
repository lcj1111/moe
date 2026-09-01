#!/usr/bin/env python3
# 作用：检查 Markdown 本地链接和仓库路径是否真实存在。
"""检查仓库文档中的本地链接和仓库内路径引用。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote


# Windows 的默认控制台编码可能不是 UTF-8；显式设置后，中文检查结果在本地和 CI 中一致。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
REPO_DIRS = {
    "analysis",
    "clients",
    "configs",
    "data",
    "docs",
    "env",
    "evaluation",
    "phase4",
    "phase7",
    "quantization",
    "runtime_patches",
    "scripts",
    "selector",
    "serving",
    "tests",
    "topology",
    "traces",
}
ROOT_FILES = {"README.md", "Makefile"}


def markdown_files(root: Path) -> list[Path]:
    """返回仓库中的全部 Markdown 文件，忽略 Git 内部目录。"""
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".git" not in path.relative_to(root).parts
    )


def normalize_link(raw: str) -> str | None:
    """把 Markdown 链接目标转成可检查的相对路径。"""
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    path = unquote(target.split("#", 1)[0])
    return path or None


def normalize_repo_reference(raw: str) -> str | None:
    """识别反引号中明确指向仓库的路径；运行时产物名不在这里猜测。"""
    value = raw.strip().replace("\\", "/").rstrip(".,;:")
    if value.startswith("./"):
        value = value[2:]
    if (
        not value
        or value.startswith(("/", "http://", "https://", "git@"))
        or " " in value
        or any(mark in value for mark in ("<", ">", "$", "*", "...", "|"))
    ):
        return None
    first = value.split("/", 1)[0]
    if first in REPO_DIRS or value in ROOT_FILES:
        return value
    return None


def check(root: Path) -> list[str]:
    """返回全部错误；调用方决定打印方式和退出码。"""
    errors: list[str] = []
    for document in markdown_files(root):
        text = document.read_text(encoding="utf-8")
        relative_document = document.relative_to(root)

        for line_number, line in enumerate(text.splitlines(), 1):
            for raw in MARKDOWN_LINK.findall(line):
                target = normalize_link(raw)
                if target is None:
                    continue
                if not (document.parent / target).resolve().exists():
                    errors.append(
                        f"{relative_document}:{line_number}: 本地链接不存在：{raw}"
                    )

            for raw in INLINE_CODE.findall(line):
                target = normalize_repo_reference(raw)
                if target is None:
                    continue
                if not (root / target).exists():
                    errors.append(
                        f"{relative_document}:{line_number}: 仓库路径不存在：{raw}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 Markdown 本地链接和仓库路径引用")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录；默认使用脚本所在仓库",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors = check(root)
    if errors:
        for error in errors:
            print(f"错误  {error}", file=sys.stderr)
        print(f"文档检查失败：共 {len(errors)} 项", file=sys.stderr)
        return 1
    print("通过  文档本地链接和仓库路径均存在")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
