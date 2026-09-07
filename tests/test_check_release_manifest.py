# 作用：验证发布文件缺失、内容变更及政策上限不一致时，检查必须失败。
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.check_release_manifest import MANIFEST, check


@pytest.fixture
def release_copy(tmp_path):
    """仅复制发布清单的直接引用，不复制环境、模型或实验日志。"""
    source = Path(__file__).resolve().parents[1]
    manifest = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    bundle = manifest["recommended_bundle"]
    paths = {MANIFEST, *(bundle[key] for key in ("model_audit", "candidate_file", "runtime_patch"))}
    for group in ("effective_policies", "authoritative_validation"):
        paths.update(item["path"] for item in manifest[group].values())
    policy = json.loads((source / manifest["effective_policies"]["selector_policy"]["path"]).read_text(encoding="utf-8"))
    paths.update(policy["验证证据"][key] for key in ("independent_gate_json", "frozen_selector_json"))
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    return tmp_path, manifest


def test_current_release_is_consistent(release_copy):
    root, _ = release_copy
    assert check(root) == []


@pytest.mark.parametrize("change", ["missing", "tamper", "escape"])
def test_bad_release_reference_fails(release_copy, change):
    root, manifest = release_copy
    report = manifest["authoritative_validation"]["final_report"]
    path = root / report["path"]
    if change == "missing":
        path.unlink()
    elif change == "tamper":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        report["path"] = "../outside.md"
        (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert check(root)


@pytest.mark.parametrize("policy,field", [
    ("selector_policy", "p95_regret_max_pct"),
    ("route_stability_policy", "placement_excess_tv_p95_max"),
])
def test_policy_threshold_mismatch_fails(release_copy, policy, field):
    root, manifest = release_copy
    manifest["effective_policies"][policy][field] = 123
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert any("上限不一致" in error for error in check(root))


def test_windows_line_endings_match_repository_hash(release_copy):
    root, manifest = release_copy
    report = manifest["authoritative_validation"]["final_report"]
    path = root / report["path"]
    data = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == report["sha256"]
    path.write_bytes(data.replace(b"\n", b"\r\n"))
    assert check(root) == []
