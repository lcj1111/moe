#!/usr/bin/env python3
"""将固定 incumbent 的决策前状态绑定到已有 Phase 8 outcome 聚合。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach(aggregate: dict[str, Any], states: dict[str, Any]) -> dict[str, Any]:
    if aggregate.get("status") != "accepted" or states.get("status") != "accepted":
        raise ValueError("outcome aggregate 或 predecision state Gate 未接受")
    state_rows = states.get("states", {})
    output_rows = []
    for row in aggregate["rows"]:
        workload_id = row["workload_id"]
        if workload_id not in state_rows:
            raise ValueError(f"{workload_id}: 缺少决策前状态")
        source = state_rows[workload_id]
        expected = {key: row.get(key) for key in (
            "input_tokens", "output_tokens", "concurrency", "prefix_cache_pct",
            "arrival_mode", "request_rate_rps")}
        if source.get("workload") != expected:
            raise ValueError(f"{workload_id}: 状态窗口与 outcome 参数不一致")
        state = source.get("selector_state", {})
        if state.get("observation_phase") != "pre_decision" or not state.get(
                "decision_eligible"):
            raise ValueError(f"{workload_id}: 状态窗口存在阶段或泄漏错误")
        output_rows.append({
            **row,
            "selector_state": state,
            "selector_state_source": {
                "incumbent_candidate_id": states["incumbent_candidate_id"],
                "summary": source["summary"],
                "summary_sha256": source["summary_sha256"],
            },
        })
    return {
        **aggregate,
        "schema_version": "qtopomoe.phase8_outcomes_with_predecision_state.v1",
        "rows": output_rows,
        "predecision_state_manifest": {
            "incumbent_candidate_id": states["incumbent_candidate_id"],
            "matrix_sha256": states["matrix_sha256"],
            "client_sha256": states["client_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--predecision-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    states = json.loads(args.predecision_states.read_text(encoding="utf-8"))
    result = attach(aggregate, states)
    result["source_artifacts"] = {
        "aggregate": str(args.aggregate),
        "aggregate_sha256": sha256(args.aggregate),
        "predecision_states": str(args.predecision_states),
        "predecision_states_sha256": sha256(args.predecision_states),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
