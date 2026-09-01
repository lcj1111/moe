#!/usr/bin/env python3
# 作用：审计 W4A16 compressed-tensors checkpoint 的覆盖完整性。
"""Static W4A16 compressed-tensors coverage audit."""
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
    manifest = json.loads((checkpoint / "qtopomoe_quant_manifest.json").read_text(encoding="utf-8"))
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    expert_packed = [k for k in keys if ".mlp.experts." in k and k.endswith(".weight_packed")]
    expert_scale = [k for k in keys if ".mlp.experts." in k and k.endswith(".weight_scale")]
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
    expected_total = expected_layers * expected_experts * expected_expert_linears
    checks = {
        "format_compressed_tensors": quant_config.get("quant_method") == "compressed-tensors" or config.get("quantization_config", {}).get("format") in {"pack-quantized", "compressed-tensors"},
        "expert_packed_count": len(expert_packed) == expected_total,
        "expert_scale_count": len(expert_scale) == expected_total,
        "layer_count": len(layer_counts) == expected_layers and set(layer_counts.values()) == {expected_experts * expected_expert_linears},
        "expert_count": len(expert_counts) == expected_experts and set(expert_counts.values()) == {expected_layers * expected_expert_linears},
        "projection_count": projections == Counter({"down_proj": expected_layers * expected_experts, "gate_proj": expected_layers * expected_experts, "up_proj": expected_layers * expected_experts}),
        "excluded_linear_attn_present": not any(".linear_attn." in k and k.endswith(".weight_packed") for k in keys),
    }
    result = {
        "schema_version": "qtopomoe.w4a16_audit.v1", "checkpoint": str(checkpoint),
        "model_safetensors_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "expected": {"layers": expected_layers, "experts": expected_experts, "expert_linears": expected_expert_linears, "expert_total": expected_total},
        "observed": {"safetensor_keys": len(keys), "expert_packed": len(expert_packed), "expert_scale": len(expert_scale), "layers": len(layer_counts), "experts": len(expert_counts), "projections": dict(projections), "excluded_regex_hits": len(excluded_hits)},
        "quantization_config": quant_config, "manifest": manifest, "checks": checks,
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
    result = audit(args.checkpoint, args.expected_layers, args.expected_experts, args.expected_expert_linears, args.excluded_regex)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "observed": result["observed"], "checks": result["checks"]}, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
