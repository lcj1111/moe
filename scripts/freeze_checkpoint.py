#!/usr/bin/env python3
"""Create a reproducible, content-addressed checkpoint freeze manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    tensor_path = checkpoint / "model.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

    prefixes = Counter()
    suffixes = Counter()
    for key in keys:
        if key.startswith("model.language_model."):
            prefixes["model.language_model"] += 1
        elif key.startswith("model."):
            prefixes["model"] += 1
        else:
            prefixes["root"] += 1
        suffixes[key.rsplit(".", 1)[-1]] += 1

    files = []
    for path in sorted(p for p in checkpoint.iterdir() if p.is_file()):
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )

    result = {
        "schema_version": "qtopomoe.checkpoint_freeze.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "files": files,
        "config_identity": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_experts": config.get("num_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "quant_method": config.get("quantization_config", {}).get("quant_method"),
            "quantization_status": config.get("quantization_config", {}).get(
                "quantization_status"
            ),
        },
        "tensor_inventory": {
            "total": len(keys),
            "prefixes": dict(prefixes),
            "suffixes": dict(suffixes),
            "first_keys": keys[:20],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "files": len(files), "tensors": len(keys)}))


if __name__ == "__main__":
    main()
