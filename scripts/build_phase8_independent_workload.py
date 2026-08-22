#!/usr/bin/env python3
"""从训练矩阵的保守速率生成参数指纹不重叠的 Phase 8 独立测试矩阵。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return sum(math.log2(float(left[key]) / float(right[key])) ** 2
               for key in ("input_tokens", "output_tokens", "concurrency"))


def stable_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def fingerprint(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in (
        "input_tokens", "output_tokens", "concurrency", "prefix_cache_pct",
        "arrival_mode", "request_rate_rps",
    ))


def build(training: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    training_cells = training["cells"]
    rate_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in training_cells:
        if row.get("arrival_mode") not in ("poisson", "burst"):
            continue
        key = (row["input_tokens"], row["output_tokens"], row["concurrency"])
        existing = rate_rows.setdefault(key, {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "concurrency": row["concurrency"],
            "request_rate_rps": row["request_rate_rps"],
        })
        if abs(float(existing["request_rate_rps"])
               - float(row["request_rate_rps"])) > 1e-9:
            raise ValueError(f"训练矩阵同形状请求率不一致：{key}")
    if not rate_rows:
        raise ValueError("训练矩阵没有可用于冻结的 open-loop 请求率")
    sources = list(rate_rows.values())
    cells = []
    rate_audit = []
    for base in design["base_cells"]:
        neighbors = sorted(sources, key=lambda row: (
            distance(base, row), row["input_tokens"], row["concurrency"]))[
                :int(design.get("rate_neighbor_count", 2))]
        source_rate = min(float(row["request_rate_rps"]) for row in neighbors)
        frozen_rate = math.floor(
            source_rate * float(design["rate_safety_multiplier"]) * 1_000_000
        ) / 1_000_000
        if frozen_rate <= 0:
            raise ValueError(f"{base['id']}: 冻结请求率无效")
        rate_audit.append({
            "base_cell_id": base["id"],
            "training_neighbors": [{
                "shape": {key: row[key] for key in (
                    "input_tokens", "output_tokens", "concurrency")},
                "request_rate_rps": row["request_rate_rps"],
                "shape_log_distance": distance(base, row),
            } for row in neighbors],
            "minimum_neighbor_rate_rps": source_rate,
            "safety_multiplier": design["rate_safety_multiplier"],
            "frozen_request_rate_rps": frozen_rate,
        })
        requests = max(int(design["minimum_requests"]),
                       int(base["concurrency"]) * 2)
        for prefix in design["prefix_cache_pct"]:
            for mode in design["arrival_modes"]:
                cell_id = f"{base['id']}_p{prefix}_{mode}"
                cell = {
                    "id": cell_id,
                    "base_cell_id": base["id"],
                    "input_tokens": base["input_tokens"],
                    "output_tokens": base["output_tokens"],
                    "concurrency": base["concurrency"],
                    "requests": requests,
                    "prefix_cache_pct": prefix,
                    "arrival_mode": mode,
                    "stream_seed": stable_seed(cell_id, int(design["seed"])),
                }
                if mode != "closed_loop":
                    cell["request_rate_rps"] = frozen_rate
                cells.append(cell)
    training_fingerprints = {fingerprint(row) for row in training_cells}
    overlaps = [row["id"] for row in cells
                if fingerprint(row) in training_fingerprints]
    if overlaps:
        raise ValueError(f"独立测试参数与训练矩阵重叠：{overlaps}")
    return {
        "schema_version": "qtopomoe.phase8_independent_workload.v1",
        "purpose": "Phase 8 冻结 selector 的完全独立测试矩阵",
        "seed": design["seed"],
        "rate_policy": {
            "source": "只读取训练矩阵；取两个最近 shape 冻结速率的较小值",
            "safety_multiplier": design["rate_safety_multiplier"],
            "rounding": "向下保留 6 位小数",
            "independent_test_outcomes_used": False,
        },
        "training_overlap_gate": True,
        "rate_audit": rate_audit,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training_matrix.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    result = build(training, design)
    result["sources"] = {
        "training_matrix": args.training_matrix.as_posix(),
        "training_matrix_sha256": sha256(args.training_matrix),
        "design": args.design.as_posix(),
        "design_sha256": sha256(args.design),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(result["cells"]),
                      "training_overlap_gate": result["training_overlap_gate"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
