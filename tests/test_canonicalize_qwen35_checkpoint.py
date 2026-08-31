# 作用：验证 checkpoint 命名空间转换不改变张量内容。
from __future__ import annotations

import importlib.util
import pathlib


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "canonicalize_qwen35_text_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("canonicalize_qwen35", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_key_mapping_changes_only_wrapper_namespace():
    assert MODULE.map_key("model.language_model.layers.0.mlp.gate.weight") == (
        "model.layers.0.mlp.gate.weight"
    )
    assert MODULE.map_key("lm_head.weight") == "lm_head.weight"
    assert MODULE.map_key("model.layers.0.input_layernorm.weight") == (
        "model.layers.0.input_layernorm.weight"
    )


def test_config_prefix_audit_reports_nested_values():
    config = {"quantization_config": {"ignore": ["model.language_model.layers.0"]}}
    found = MODULE.find_old_prefixes(config)
    assert len(found) == 1
    assert "quantization_config.ignore[0]" in found[0]
