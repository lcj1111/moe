#!/usr/bin/env python3
"""Static NVFP4 (compressed-tensors, nvfp4-pack-quantized) coverage audit.

NVFP4 checkpoints produced by llmcompressor's
QuantizationModifier(scheme="NVFP4") keep per-projection packed weights:
  <prefix>.mlp.experts.<e>.<down|gate|up>_proj.weight_packed
  <prefix>.mlp.experts.<e>.<proj>.weight_scale / weight_global_scale
  <prefix>.mlp.experts.<e>.<proj>.input_global_scale
The VLM wrapper carries a model.language_model.* prefix which is stripped for
parsing.  Coverage baseline: 40 layers x 256 routed experts x 3 projections
= 30720 packed expert tensors (same as W4A16).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def audit(checkpoint: Path, expected_layers: int, expected_experts: int,
          expected_expert_linears: int, excluded_regex: str) -> dict:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"model.safetensors missing in {checkpoint}")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

    def norm(k: str) -> str:
        return k.replace("model.language_model.", "", 1)

    expert_packed = [norm(k) for k in keys
                     if ".mlp.experts." in k and k.endswith("_proj.weight_packed")]
    expert_scale = [norm(k) for k in keys
                    if ".mlp.experts." in k and k.endswith("_proj.weight_scale")]
    expert_global_scale = [norm(k) for k in keys
                           if ".mlp.experts." in k and k.endswith("_proj.weight_global_scale")]
    expert_input_scale = [norm(k) for k in keys
                          if ".mlp.experts." in k and k.endswith("_proj.input_global_scale")]
    layer_counts: Counter[int] = Counter()
    expert_counts: Counter[int] = Counter()
    projections: Counter[str] = Counter()
    for key in expert_packed:
        parts = key.split(".")
        layer = int(parts[parts.index("layers") + 1])
        expert = int(parts[parts.index("experts") + 1])
        projection = parts[-2]
        layer_counts[layer] += 1
        expert_counts[expert] += 1
        projections[projection] += 1
    excluded_hits = sorted(k for k in keys if re.search(excluded_regex, k))

    quant_config = config.get("quantization_config", {})
    group0 = quant_config.get("config_groups", {}).get("group_0", {})
    fmt = group0.get("format", "")
    expected_total = expected_layers * expected_experts * expected_expert_linears
    checks = {
        "format_nvfp4_pack_quantized": "nvfp4" in str(fmt).lower() and "pack" in str(fmt).lower(),
        "weights_num_bits_4": group0.get("weights", {}).get("num_bits") == 4,
        "activations_num_bits_4": group0.get("input_activations", {}).get("num_bits") == 4,
        "expert_packed_count": len(expert_packed) == expected_total,
        "expert_scale_count": len(expert_scale) == expected_total,
        "expert_global_scale_count": len(expert_global_scale) == expected_total,
        "expert_input_scale_count": len(expert_input_scale) == expected_total,
        "layer_count": len(layer_counts) == expected_layers
        and set(layer_counts.values()) == {expected_experts * expected_expert_linears},
        "expert_count": len(expert_counts) == expected_experts
        and set(expert_counts.values()) == {expected_layers * expected_expert_linears},
        "projection_count": projections == Counter({
            "down_proj": expected_layers * expected_experts,
            "gate_proj": expected_layers * expected_experts,
            "up_proj": expected_layers * expected_experts,
        }),
        "excluded_linear_attn_absent": not any(
            ".linear_attn." in norm(k) and k.endswith("_proj.weight_packed") for k in keys
        ),
    }
    result = {
        "schema_version": "qtopomoe.nvfp4_audit.v1",
        "checkpoint": str(checkpoint),
        "model_safetensors_bytes": weights.stat().st_size,
        "expected": {"layers": expected_layers, "experts": expected_experts,
                     "expert_linears": expected_expert_linears,
                     "expert_total": expected_total},
        "observed": {"safetensor_keys": len(keys), "expert_packed": len(expert_packed),
                     "expert_scale": len(expert_scale),
                     "expert_global_scale": len(expert_global_scale),
                     "expert_input_scale": len(expert_input_scale),
                     "layers": len(layer_counts), "experts": len(expert_counts),
                     "projections": dict(projections),
                     "excluded_regex_hits": len(excluded_hits)},
        "quantization_config": quant_config,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--expected-experts", type=int, default=256)
    parser.add_argument("--expected-expert-linears", type=int, default=3)
    parser.add_argument("--excluded-regex", default=r".*linear_attn.*")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.expected_layers, args.expected_experts,
                   args.expected_expert_linears, args.excluded_regex)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "observed": result["observed"], "checks": result["checks"]}, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
