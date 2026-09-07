#!/usr/bin/env python3
# 作用：检查发布入口、仓库文件哈希、政策上限及 selector 的直接证据引用。
"""校验工作区内容；文本按仓库 .gitattributes 的 LF 口径计算 SHA-256。

不校验服务器绝对路径、模型权重或未入库的 Gate 文件，仅检查明确列出的仓库证据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MANIFEST = "docs/Q-TopoMoE_release_manifest_20260825.json"


def check(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []

    def read(relative: str, expected: str | None = None) -> bytes:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"发布文件不存在或超出仓库：{relative}")
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if expected is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                errors.append(f"SHA-256 格式错误：{relative}")
            elif hashlib.sha256(data).hexdigest() != expected.lower():
                errors.append(f"SHA-256 不匹配：{relative}")
        return data

    def verify(relative: str, expected: str) -> None:
        try:
            read(relative, expected)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    try:
        manifest = json.loads(read(MANIFEST))
        if manifest["release_branch"] != "main":
            errors.append("当前 release_branch 必须为 main")
        if not manifest["release_tag"] or not manifest["historical_release_branch"]:
            errors.append("缺少历史发布标签或来源分支")
        bundle = manifest["recommended_bundle"]
        for key in ("model_audit", "candidate_file", "runtime_patch"):
            verify(bundle[key], bundle[key + "_sha256"])
        for group in ("effective_policies", "authoritative_validation"):
            for item in manifest[group].values():
                verify(item["path"], item["sha256"])

        policy = json.loads(read(manifest["effective_policies"]["selector_policy"]["path"]))
        evidence = policy["验证证据"]
        verify(evidence["independent_gate_json"], evidence["independent_gate_sha256"])
        verify(evidence["frozen_selector_json"], evidence["repository_frozen_selector_sha256"])
        if (policy["准入门槛"]["p95_regret_pct_max"] !=
                manifest["effective_policies"]["selector_policy"]["p95_regret_max_pct"]):
            errors.append("selector 政策与发布清单上限不一致")
        route = manifest["effective_policies"]["route_stability_policy"]
        route_policy = json.loads(read(route["path"]))
        for trajectory in ("short", "long"):
            if (route_policy["当前有效门槛"][trajectory + "_placement_excess_tv_p95_max"] !=
                    route["placement_excess_tv_p95_max"]):
                errors.append(f"{trajectory} 路由政策与发布清单上限不一致")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"发布清单结构或引用错误：{exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    errors = check(parser.parse_args().root)
    for error in errors:
        print(f"FAIL {error}")
    if not errors:
        print("PASS 发布入口、清单文件哈希与 selector 直接证据引用")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
