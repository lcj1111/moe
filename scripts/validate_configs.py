#!/usr/bin/env python3
# 作用：按最小结构约束校验项目 YAML 与 JSON 配置。
"""检查配置能否解析、顶层是否为字典或列表；不替代各实验入口的字段与语义校验。"""
import json
from pathlib import Path
import sys

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(path for path in (root / "configs").rglob("*")
                   if path.suffix in {".yaml", ".yml", ".json"})
    if not paths:
        print("FAIL no configs found", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle) if path.suffix == ".json" else yaml.safe_load(handle)
            if not isinstance(data, (dict, list)):
                raise ValueError("top-level value must be a mapping or list")
            print(f"PASS  {path.suffix[1:].upper()} {path.relative_to(root)}")
        except Exception as exc:
            failed = True
            print(f"FAIL  {path.suffix[1:].upper()} {path.relative_to(root)}: {exc}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
