#!/usr/bin/env python3
"""针对固定 incumbent 服务采集 Phase 8 决策前状态窗口。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def health_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def audit(summary: dict[str, Any], cell: dict[str, Any]) -> None:
    state = summary.get("selector_state", {})
    service = summary.get("service_telemetry", {})
    checks = {
        "failed_zero": summary.get("failed") == 0,
        "complete": summary.get("completed") == summary.get("requests"),
        "input_tokens": summary.get("input_tokens_actual") == cell["input_tokens"],
        "concurrency": summary.get("concurrency") == cell["concurrency"],
        "cache_target": summary.get("prefix_cache", {}).get("target_pct")
                        == cell.get("prefix_cache_pct", 0),
        "cache_ratio": summary.get("prefix_cache", {}).get("ratio_gate") is True,
        "arrival_mode": summary.get("arrival", {}).get("mode")
                        == cell.get("arrival_mode", "closed_loop"),
        "arrival_schedule": summary.get("arrival", {}).get("schedule_gate") is True,
        "pre_decision": state.get("observation_phase") == "pre_decision"
                        and state.get("decision_eligible") is True,
        "server_queue": state.get("server_queue_depth_available") is True,
        "metrics_coverage": isinstance(service.get("coverage_ratio"), (int, float))
                            and service["coverage_ratio"] >= 0.9,
    }
    if not all(checks.values()):
        raise RuntimeError(f"决策前状态窗口 Gate 失败：{cell['id']} {checks}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--incumbent-candidate-id", required=True)
    parser.add_argument("--cache-block-tokens", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--selector-window-requests", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not health_ready(args.base_url):
        raise SystemExit("incumbent 服务未就绪")
    repo = Path(__file__).resolve().parents[1]
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "qtopomoe.phase8_predecision_run.v1",
        "status": "running",
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": sha256(args.matrix),
        "incumbent_candidate_id": args.incumbent_candidate_id,
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "cache_block_tokens": args.cache_block_tokens,
        "selector_window_requests": args.selector_window_requests,
        "client_sha256": sha256(repo / "clients" / "smoke.py"),
        "completed_cells": [],
    }
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("matrix_sha256", "incumbent_candidate_id", "model", "tokenizer",
                    "cache_block_tokens", "selector_window_requests", "client_sha256"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"已有 manifest 与本次 {key} 不一致")
        manifest["completed_cells"] = existing.get("completed_cells", [])
    write_json(manifest_path, manifest)

    states: dict[str, Any] = {}
    for index, cell in enumerate(matrix["cells"]):
        cell_dir = args.output_root / "cells" / cell["id"]
        summary_path = cell_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            audit(summary, cell)
        else:
            cell_dir.mkdir(parents=True, exist_ok=True)
            requests = max(
                int(cell["concurrency"]),
                min(int(cell["requests"]), args.selector_window_requests),
            )
            command = [
                args.python_bin, str(repo / "clients" / "smoke.py"),
                "--base-url", args.base_url, "--model", args.model,
                "--tokenizer", args.tokenizer,
                "--input-tokens", str(cell["input_tokens"]),
                "--output-tokens", str(cell["output_tokens"]),
                "--concurrency", str(cell["concurrency"]),
                "--requests", str(requests),
                "--seed", str(920000 + index),
                "--stream-seed", str(cell.get("stream_seed", 920000 + index)),
                "--prefix-cache-pct", str(cell.get("prefix_cache_pct", 0)),
                "--arrival-mode", str(cell.get("arrival_mode", "closed_loop")),
                "--cache-block-tokens", str(args.cache_block_tokens),
                "--require-cache-details", "--require-arrival-gate",
                "--sample-service-metrics", "--metrics-sample-interval-s", "0.2",
                "--observation-phase", "pre_decision",
                "--selector-window-requests", str(args.selector_window_requests),
                "--timeout", str(args.timeout),
                "--output", str(cell_dir / "requests.jsonl"),
                "--summary", str(summary_path),
            ]
            if cell.get("arrival_mode", "closed_loop") != "closed_loop":
                command.extend(["--request-rate", str(cell["request_rate_rps"])])
            subprocess.run(command, cwd=repo, check=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            audit(summary, cell)
        states[cell["id"]] = {
            "workload": {key: cell.get(key) for key in (
                "input_tokens", "output_tokens", "concurrency", "prefix_cache_pct",
                "arrival_mode", "request_rate_rps")},
            "selector_state": summary["selector_state"],
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
        }
        if cell["id"] not in manifest["completed_cells"]:
            manifest["completed_cells"].append(cell["id"])
            write_json(manifest_path, manifest)
    manifest["status"] = "accepted"
    manifest["cell_count"] = len(states)
    write_json(manifest_path, manifest)
    bundle = {**manifest, "states": states}
    write_json(args.output_root / "predecision_states.json", bundle)
    print(json.dumps({"status": "accepted", "cell_count": len(states)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
