#!/usr/bin/env python3
# 作用：读取 safetensors 元数据并统计量化专家的实际存储字节。
"""Read safetensors metadata and report exact routed-expert storage bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.checkpoint)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    expected_layers = int(text_config["num_hidden_layers"])
    expected_experts = int(text_config["num_experts"])

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = sorted(path.name for path in root.glob("*.safetensors"))
    if not shard_names:
        raise SystemExit("no safetensors checkpoint files found")

    per_expert: dict[str, int] = defaultdict(int)
    tensor_counts: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    routed_tensor_count = 0
    for shard_name in shard_names:
        with safe_open(root / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                match = EXPERT_RE.search(name)
                if not match:
                    continue
                layer, expert = map(int, match.groups())
                tensor_slice = handle.get_slice(name)
                dtype = tensor_slice.get_dtype()
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
                elements = math.prod(tensor_slice.get_shape())
                size = elements * DTYPE_BYTES[dtype]
                key = f"{layer}:{expert}"
                per_expert[key] += size
                tensor_counts[key] += 1
                dtype_bytes[dtype] += size
                routed_tensor_count += 1

    expected_count = expected_layers * expected_experts
    missing = [
        f"{layer}:{expert}"
        for layer in range(expected_layers)
        for expert in range(expected_experts)
        if f"{layer}:{expert}" not in per_expert
    ]
    unique_sizes = sorted(set(per_expert.values()))
    result = {
        "schema_version": "qtopomoe.quantized_expert_size_audit.v1",
        "checkpoint": str(root),
        "quant_format": config.get("quantization_config", {}).get("format", "unknown"),
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path) if index_path.exists() else None,
        "expected_layers": expected_layers,
        "expected_experts_per_layer": expected_experts,
        "expected_expert_count": expected_count,
        "observed_expert_count": len(per_expert),
        "routed_tensor_count": routed_tensor_count,
        "missing_experts": missing,
        "uniform_expert_bytes": unique_sizes[0] if len(unique_sizes) == 1 else None,
        "unique_expert_sizes_bytes": unique_sizes,
        "total_routed_expert_bytes": sum(per_expert.values()),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "tensor_count_values": sorted(set(tensor_counts.values())),
        "per_expert_bytes": dict(sorted(per_expert.items(), key=lambda item: tuple(map(int, item[0].split(":"))))),
        "accepted": len(per_expert) == expected_count and not missing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "accepted", "observed_expert_count", "routed_tensor_count",
        "uniform_expert_bytes", "total_routed_expert_bytes", "dtype_bytes"
    )}, indent=2))


if __name__ == "__main__":
    main()
