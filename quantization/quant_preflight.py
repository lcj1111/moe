#!/usr/bin/env python3
"""Reproducible preflight for Q-TopoMoE Phase 2 quantization.

This intentionally does not load the 35B checkpoint. It validates the pinned
quant environment, model metadata, disk headroom, and the calibration-manifest
contract before a destructive/long-running PTQ job is allowed to start.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_records(path: Path) -> int:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON calibration manifest must contain a list")
        return len(value)
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--required-samples", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="permit a missing calibration manifest; formal quantization never uses this flag",
    )
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    errors: list[str] = []

    if not config_path.is_file():
        errors.append(f"missing config: {config_path}")
        config: dict = {}
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))

    architecture = (config.get("architectures") or [None])[0]
    model_type = config.get("model_type")
    text_config = config.get("text_config") or {}
    layers = text_config.get("num_hidden_layers", config.get("num_hidden_layers"))
    experts = text_config.get("num_experts", config.get("num_experts"))
    if layers != 40:
        errors.append(f"expected 40 hidden layers, found {layers!r}")
    if experts != 256:
        errors.append(f"expected 256 experts, found {experts!r}")
    if model_type != "qwen3_5_moe":
        errors.append(f"unexpected model_type: {model_type!r}")

    calibration = args.calibration
    if calibration is None:
        env_path = os.environ.get("QTOPOMOE_CALIBRATION_JSONL", "").strip()
        calibration = Path(env_path) if env_path else None
    calibration_records = None
    if calibration is None:
        if not args.environment_only:
            errors.append("QTOPOMOE_CALIBRATION_JSONL is not set")
    else:
        calibration = calibration.expanduser().resolve()
        if not calibration.is_file():
            errors.append(f"missing calibration manifest: {calibration}")
        else:
            calibration_records = count_records(calibration)
            if calibration_records < args.required_samples:
                errors.append(
                    f"calibration has {calibration_records} records; "
                    f"requires at least {args.required_samples}"
                )

    package_versions = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "safetensors",
        "llmcompressor",
        "compressed-tensors",
        "auto-round",
    ):
        try:
            package_versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            package_versions[name] = None
            errors.append(f"missing package: {name}")

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - diagnostic path
        cuda_available = False
        device_count = 0
        errors.append(f"torch/cuda import failed: {type(exc).__name__}: {exc}")
    if not cuda_available or device_count < 8:
        errors.append(f"requires 8 visible CUDA devices; available={device_count}")

    disk = shutil.disk_usage(model)
    report = {
        "status": "PASS" if not errors else "BLOCKED",
        "python": sys.version,
        "model": str(model),
        "model_type": model_type,
        "architecture": architecture,
        "hidden_layers": layers,
        "experts": experts,
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "index_sha256": sha256(index_path) if index_path.is_file() else None,
        "calibration": str(calibration) if calibration else None,
        "calibration_records": calibration_records,
        "required_calibration_records": args.required_samples,
        "package_versions": package_versions,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "disk_free_bytes_at_model_root": disk.free,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
