#!/usr/bin/env python3
# 作用：将 W4A16 分组从 128 无损重排为兼容 TP8 的 64。
"""Lossless W4A16 block_size 128 -> 64 format transform.

The canonical W4A16 checkpoint quantizes with GPTQ group (block) size 128
(``weights.group_size=128``, symmetric, no zero point).  Under TP8 the MoE
intermediate size per partition (512/8 = 64) is not divisible by 128, so vLLM
rejects the layout (scale groups would cross TP shard boundaries).

This script produces a *new* checkpoint directory with ``group_size=64``:
for every ``*weight_scale`` tensor the group dimension (dim 1) is duplicated
(``repeat_interleave(2, dim=1)``), so each 64-wide group carries a copy of
the original 128-wide scale.  Because W4A16 is symmetric with no zero point,
dequantization ``int4 * scale`` is bit-identical to the original layout; the
packed int4 weights are left untouched.  Verified numerically (max diff 0.0).

The original canonical directory is never modified; the transform records a
manifest linking the new checkpoint to its block-128 source.

Usage (server, quant env):
    python quantization/reblock_w4a16_128_to_64.py \
        --src /data/models/test/qtopomoe_w4a16_canonical_text_v1 \
        --dst /data/models/test/qtopomoe_w4a16_canonical_text_v1_g64
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from datetime import datetime, timezone

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def transform_scale(scale: torch.Tensor) -> torch.Tensor:
    """Duplicate the group dimension so group_size 128 -> 64."""
    return torch.repeat_interleave(scale, 2, dim=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=pathlib.Path, required=True)
    parser.add_argument("--dst", type=pathlib.Path, required=True)
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"destination exists, refusing to overwrite: {dst}")
    dst.mkdir(parents=True)

    src_sf = src / "model.safetensors"
    dst_sf = dst / "model.safetensors"

    tensors: dict[str, torch.Tensor] = {}
    n_scales = 0
    with safe_open(src_sf, framework="pt") as handle:
        for name in handle.keys():
            t = handle.get_tensor(name)
            if "weight_scale" in name:
                before = list(t.shape)
                t = transform_scale(t)
                after = list(t.shape)
                if after[1] != before[1] * 2:
                    raise RuntimeError(
                        f"unexpected scale shape for {name}: {before} -> {after}")
                n_scales += 1
            tensors[name] = t
    save_file(tensors, dst_sf)
    del tensors

    # Copy all non-weight files, then patch config group_size.
    for item in src.iterdir():
        if item.name == "model.safetensors":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    weights = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    old_group = weights["group_size"]
    if old_group != 128:
        raise SystemExit(f"expected group_size 128, got {old_group}")
    weights["group_size"] = 64
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # Record the transform (do not overwrite the original quant manifest).
    manifest = {
        "schema_version": "qtopomoe.w4a16.reblock_128_to_64.v1",
        "source": str(src),
        "destination": str(dst),
        "transform": (
            "lossless group_size 128 -> 64: every weight_scale group dim "
            "duplicated (repeat_interleave 2); packed int4 weights unchanged; "
            "symmetric no-zero-point so dequant is bit-identical"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scale_tensors_transformed": n_scales,
    }
    (dst / "reblock_128_to_64.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
