#!/usr/bin/env python3
"""Create a value-identical, text-only Qwen3.5 serving checkpoint.

The frozen source checkpoint has a text-only config but retains the VLM wrapper
namespace ``model.language_model.*``.  Current text-only vLLM and SGLang model
classes expect ``model.*``.  This script changes keys only, never tensor values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OLD_PREFIX = "model.language_model."
NEW_PREFIX = "model."
WEIGHTS = "model.safetensors"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_old_prefixes(value: Any, location: str = "config") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and OLD_PREFIX in value:
        found.append(f"{location}={value}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_old_prefixes(item, f"{location}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(find_old_prefixes(item, f"{location}.{key}"))
    return found


def map_key(key: str) -> str:
    if key.startswith(OLD_PREFIX):
        return NEW_PREFIX + key[len(OLD_PREFIX) :]
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    source_weights = source / WEIGHTS
    source_config = source / "config.json"

    if not source.is_dir() or not source_weights.is_file() or not source_config.is_file():
        raise SystemExit("source must contain config.json and model.safetensors")
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    if source == destination or source in destination.parents:
        raise SystemExit("destination must be a separate sibling checkpoint directory")

    source_sha = sha256_file(source_weights)
    if source_sha != args.expected_source_sha256.lower():
        raise SystemExit(
            f"source SHA-256 mismatch: expected {args.expected_source_sha256}, got {source_sha}"
        )

    config = json.loads(source_config.read_text())
    if config.get("architectures") != ["Qwen3_5MoeForCausalLM"]:
        raise SystemExit(f"unexpected architectures: {config.get('architectures')!r}")
    if config.get("model_type") != "qwen3_5_moe_text":
        raise SystemExit(f"unexpected model_type: {config.get('model_type')!r}")
    config_old_prefixes = find_old_prefixes(config)
    if config_old_prefixes:
        raise SystemExit(
            "config still contains wrapper namespace; audit before conversion: "
            + "; ".join(config_old_prefixes[:10])
        )

    staging = destination.parent / f".{destination.name}.partial-{os.getpid()}"
    if staging.exists():
        raise SystemExit(f"staging path already exists: {staging}")
    staging.mkdir(parents=False)

    try:
        for entry in source.iterdir():
            if entry.name == WEIGHTS:
                continue
            if entry.is_file():
                shutil.copy2(entry, staging / entry.name)

        tensors: dict[str, torch.Tensor] = {}
        reverse_mapping: dict[str, str] = {}
        with safe_open(source_weights, framework="pt", device="cpu") as handle:
            source_keys = list(handle.keys())
            metadata = handle.metadata()
            for old_key in source_keys:
                new_key = map_key(old_key)
                if new_key in tensors:
                    raise RuntimeError(
                        f"key collision: {old_key!r} and {reverse_mapping[new_key]!r} -> {new_key!r}"
                    )
                tensors[new_key] = handle.get_tensor(old_key)
                reverse_mapping[new_key] = old_key

        mapped_count = sum(key.startswith(OLD_PREFIX) for key in source_keys)
        if mapped_count != len(source_keys) - 1:
            raise RuntimeError(
                f"expected all but lm_head to use {OLD_PREFIX!r}; "
                f"mapped={mapped_count}, total={len(source_keys)}"
            )
        if set(source_keys) - {"lm_head.weight"} != {
            key for key in source_keys if key.startswith(OLD_PREFIX)
        }:
            raise RuntimeError("unexpected root tensor keys in source checkpoint")
        if any(key.startswith(OLD_PREFIX) for key in tensors):
            raise RuntimeError("old wrapper namespace remains after mapping")

        destination_weights = staging / WEIGHTS
        save_file(tensors, destination_weights, metadata=metadata)

        verified_tensor_bytes = 0
        with safe_open(source_weights, framework="pt", device="cpu") as src, safe_open(
            destination_weights, framework="pt", device="cpu"
        ) as dst:
            if set(dst.keys()) != set(tensors):
                raise RuntimeError("destination tensor key set differs from planned key set")
            for new_key, old_key in reverse_mapping.items():
                source_tensor = src.get_tensor(old_key)
                destination_tensor = dst.get_tensor(new_key)
                if source_tensor.shape != destination_tensor.shape:
                    raise RuntimeError(f"shape mismatch for {old_key} -> {new_key}")
                if source_tensor.dtype != destination_tensor.dtype:
                    raise RuntimeError(f"dtype mismatch for {old_key} -> {new_key}")
                if not torch.equal(source_tensor, destination_tensor):
                    raise RuntimeError(f"value mismatch for {old_key} -> {new_key}")
                verified_tensor_bytes += source_tensor.numel() * source_tensor.element_size()

        destination_sha = sha256_file(destination_weights)
        files = []
        for path in sorted(staging.iterdir()):
            if path.is_file() and path.name != "canonicalization_manifest.json":
                files.append(
                    {"name": path.name, "size_bytes": path.stat().st_size,
                     "sha256": sha256_file(path)}
                )
        manifest = {
            "schema_version": "qtopomoe.qwen35_text_canonicalization.v1",
            "source": str(source),
            "destination": str(destination),
            "mapping_rule": f"{OLD_PREFIX}* -> {NEW_PREFIX}*",
            "unchanged_root_keys": ["lm_head.weight"],
            "source_weight_sha256": source_sha,
            "destination_weight_sha256": destination_sha,
            "source_tensor_count": len(source_keys),
            "destination_tensor_count": len(tensors),
            "mapped_tensor_count": mapped_count,
            "verified_tensor_bytes": verified_tensor_bytes,
            "shape_dtype_value_equality": True,
            "config_changed": False,
            "files": files,
        }
        (staging / "canonicalization_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        staging.rename(destination)
        print(json.dumps(manifest, ensure_ascii=False))
    except Exception:
        print(f"conversion failed; staging preserved for audit: {staging}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
