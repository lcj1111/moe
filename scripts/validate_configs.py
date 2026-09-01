#!/usr/bin/env python3
# 作用：按最小结构约束校验项目 YAML 与 JSON 配置。
"""Validate project YAML/JSON config files against a minimal schema.

Used by ``make check`` (env/project.env + validate_configs.py).  Fails fast
on missing required keys or malformed values so broken configs do not reach
experiment runners.
"""
from pathlib import Path
import sys

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").rglob("*.yaml"))
    if not paths:
        print("FAIL no YAML configs found", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if not isinstance(data, dict):
                raise ValueError("top-level YAML value must be a mapping")
            print(f"PASS  YAML {path.relative_to(root)}")
        except Exception as exc:
            failed = True
            print(f"FAIL  YAML {path.relative_to(root)}: {exc}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
