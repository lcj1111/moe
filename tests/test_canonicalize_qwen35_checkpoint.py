# 作用：验证 checkpoint 命名空间转换不改变张量内容。
from __future__ import annotations

import importlib.util
import pathlib
import hashlib
import json
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file


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


def test_small_checkpoint_preserves_tensor_values(tmp_path):
    """用真实小张量执行转换，验证名称、类型、数值和源文件保护。"""
    source = tmp_path / "source"
    destination = tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
    }), encoding="utf-8")
    original = {
        "model.language_model.layers.0.mlp.gate.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "lm_head.weight": torch.ones((2, 4), dtype=torch.bfloat16),
    }
    weights = source / "model.safetensors"
    save_file(original, weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    command = [
        sys.executable, str(SCRIPT), "--source", str(source),
        "--destination", str(destination), "--expected-source-sha256", digest,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    converted = load_file(destination / "model.safetensors")
    assert set(converted) == {MODULE.map_key(key) for key in original}
    for key, tensor in original.items():
        result = converted[MODULE.map_key(key)]
        assert result.dtype == tensor.dtype
        assert torch.equal(result, tensor)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == digest
    assert subprocess.run(command, capture_output=True).returncode != 0
