#!/usr/bin/env python3
# 作用：比较参考模型与候选模型的逐层专家路由差异并输出漂移指标。
"""Compare two route-trace captures (reference vs candidate).

Metrics are computed from expert IDs only (the vLLM-native capture does
not export router probabilities):
  - per-layer mean token-level top-k Jaccard;
  - per-layer assignment flip rate;
  - per-layer aggregate expert-count Pearson correlation;
  - per-layer count-based load CV (std/mean) for each model;
  - M-bucket distribution delta between the two histograms.

Router-probability KL is intentionally absent: ``topk_weights`` are not
exported by the capture mechanism and are recorded as ``null``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np


def load_capture(directory: pathlib.Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    manifest = json.loads((directory / "capture_manifest.json").read_text())
    files = sorted((directory / "traces").glob("*.npy"))
    arrays = [np.load(path) for path in files]
    return manifest, arrays


def count_cv(counts: dict[int, int]) -> float | None:
    values = list(counts.values())
    if not values:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    ref_manifest, ref_arrays = load_capture(args.reference_dir)
    cand_manifest, cand_arrays = load_capture(args.candidate_dir)

    ref_prompts = {p["prompt_id"]: p for p in ref_manifest["prompts"]}
    cand_prompts = {p["prompt_id"]: p for p in cand_manifest["prompts"]}
    if list(ref_prompts) != list(cand_prompts):
        raise SystemExit("prompt id order mismatch between captures")
    if any(
        ref_prompts[k]["prompt_token_ids_sha256"]
        != cand_prompts[k]["prompt_token_ids_sha256"]
        for k in ref_prompts
    ):
        raise SystemExit("prompt token-id hashes differ between captures")

    ref_all = np.concatenate(ref_arrays, axis=0)
    cand_all = np.concatenate(cand_arrays, axis=0)
    if ref_all.shape != cand_all.shape:
        raise SystemExit(f"shape mismatch: {ref_all.shape} vs {cand_all.shape}")

    num_layers = int(ref_all.shape[1])
    top_k = int(ref_all.shape[2])
    num_tokens = int(ref_all.shape[0])

    jaccard: list[float] = []
    flip_rate: list[float] = []
    corr: list[float] = []
    ref_cv: list[float | None] = []
    cand_cv: list[float | None] = []
    for layer in range(num_layers):
        a = ref_all[:, layer, :]
        b = cand_all[:, layer, :]
        # token-level top-k Jaccard and flip rate
        jac_sum = 0.0
        flip_sum = 0.0
        for t in range(num_tokens):
            set_a = set(a[t].tolist())
            set_b = set(b[t].tolist())
            union = set_a | set_b
            jac_sum += len(set_a & set_b) / len(union)
            flip_sum += 1.0 if set_a != set_b else 0.0
        jaccard.append(jac_sum / num_tokens)
        flip_rate.append(flip_sum / num_tokens)

        # aggregate per-expert counts
        ref_counts = defaultdict(int)
        cand_counts = defaultdict(int)
        for expert in a.reshape(-1).tolist():
            ref_counts[int(expert)] += 1
        for expert in b.reshape(-1).tolist():
            cand_counts[int(expert)] += 1
        all_ids = sorted(set(ref_counts) | set(cand_counts))
        r = np.array([ref_counts.get(i, 0) for i in all_ids], dtype=np.float64)
        c = np.array([cand_counts.get(i, 0) for i in all_ids], dtype=np.float64)
        denom = math.sqrt(float((r * r).sum()) * float((c * c).sum()))
        corr.append(float((r * c).sum() / denom) if denom > 0 else 1.0)
        ref_cv.append(count_cv(dict(ref_counts)))
        cand_cv.append(count_cv(dict(cand_counts)))

    ref_hist = json.loads((args.reference_dir / "expert_token_histogram.json").read_text())
    cand_hist = json.loads((args.candidate_dir / "expert_token_histogram.json").read_text())
    all_buckets = sorted(set(ref_hist["m_buckets"]) | set(cand_hist["m_buckets"]))
    bucket_delta = {
        bucket: cand_hist["m_buckets"].get(bucket, 0) - ref_hist["m_buckets"].get(bucket, 0)
        for bucket in all_buckets
    }

    summary = {
        "schema_version": "qtopomoe.route_drift.v1",
        "reference_model": ref_manifest["model_id"],
        "candidate_model": cand_manifest["model_id"],
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "top_k": top_k,
        "prompt_token_ids_match": True,
        "metrics": {
            "per_layer": [
                {
                    "layer_id": layer,
                    "mean_jaccard": round(jaccard[layer], 6),
                    "flip_rate": round(flip_rate[layer], 6),
                    "count_correlation": round(corr[layer], 6),
                    "ref_count_cv": round(ref_cv[layer], 6) if ref_cv[layer] else None,
                    "cand_count_cv": round(cand_cv[layer], 6) if cand_cv[layer] else None,
                }
                for layer in range(num_layers)
            ],
            "mean_jaccard": round(float(np.mean(jaccard)), 6),
            "mean_flip_rate": round(float(np.mean(flip_rate)), 6),
            "mean_count_correlation": round(float(np.mean(corr)), 6),
            "m_bucket_delta": bucket_delta,
        },
        "note": (
            "Expert-ID-only metrics. Router-probability KL requires "
            "topk_weights, which the native capture does not export."
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "tokens": num_tokens,
                "mean_jaccard": summary["metrics"]["mean_jaccard"],
                "mean_flip_rate": summary["metrics"]["mean_flip_rate"],
                "mean_count_correlation": summary["metrics"]["mean_count_correlation"],
                "m_bucket_delta": bucket_delta,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
