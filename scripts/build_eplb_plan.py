#!/usr/bin/env python3
# 作用：根据路由与专家大小生成可审计 EPLB 放置计划。
"""Build an auditable EPLB placement plan from trace and size manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selector.eplb_policy import (
    OfflineEPLBConfig,
    OfflineExpertPlacement,
    QuantizedExpertProfile,
    TopologyCostMatrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histogram", required=True)
    parser.add_argument("--size-audit", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    histogram_path = Path(args.histogram)
    size_path = Path(args.size_audit)
    topology_path = Path(args.topology)
    histogram = json.loads(histogram_path.read_text(encoding="utf-8"))
    size_audit = json.loads(size_path.read_text(encoding="utf-8"))
    cfg = json.loads(topology_path.read_text(encoding="utf-8"))
    if not size_audit.get("accepted"):
        raise SystemExit("expert size audit is not accepted")

    topology = TopologyCostMatrix(
        gpus=tuple(cfg["gpus"]),
        numa_by_gpu={int(k): int(v) for k, v in cfg["numa_by_gpu"].items()},
        pair_us_per_gb=cfg["pair_us_per_gb"],
        default_us_per_gb=float(cfg["default_us_per_gb"]),
    )
    placement_cfg = OfflineEPLBConfig(**cfg.get("placement_policy", {}))
    planner = OfflineExpertPlacement(
        topology,
        {int(k): int(v) for k, v in cfg["gpu_capacity_bytes"].items()},
        {int(k): int(v) for k, v in cfg["hbm_headroom_bytes"].items()},
        placement_cfg,
    )

    per_expert_bytes = size_audit["per_expert_bytes"]
    service_time = float(cfg["service_time_us_per_token"])
    activation_bytes = int(cfg["activation_bytes_per_token"])
    quant_format = size_audit["quant_format"]
    experts = []
    counts_by_layer = {
        int(layer["layer_id"]): {int(k): int(v) for k, v in layer["expert_counts"].items()}
        for layer in histogram["per_layer"]
    }
    expected_layers = int(size_audit["expected_layers"])
    expected_experts = int(size_audit["expected_experts_per_layer"])
    for layer_id in range(expected_layers):
        counts = counts_by_layer.get(layer_id, {})
        for expert_id in range(expected_experts):
            count = counts.get(expert_id, 0)
            key = f"{layer_id}:{expert_id}"
            experts.append(QuantizedExpertProfile(
                layer_id=layer_id,
                expert_id=expert_id,
                token_load=float(count),
                service_time_us_per_token=service_time,
                size_bytes=int(per_expert_bytes[key]),
                activation_bytes_per_token=activation_bytes,
                quant_format=quant_format,
            ))

    plan = planner.plan(experts, args.trace_sha256)
    offline_inputs_ready = (
        cfg["service_time_source"] == "measured"
        and cfg["hbm_budget_source"] == "measured_ep_load_gate"
    )
    blocking_gates = []
    if not offline_inputs_ready:
        blocking_gates.append("measured offline service-time/HBM inputs")
    if cfg.get("migration_service_gate") != "accepted":
        blocking_gates.append("live-service migration block/recovery/p99")
    if cfg.get("runtime_plan_application_gate") != "accepted":
        blocking_gates.append("runtime placement-plan application")
    result = {
        "schema_version": "qtopomoe.eplb_placement.v1",
        "inputs": {
            "histogram": str(histogram_path),
            "size_audit": str(size_path),
            "topology": str(topology_path),
            "service_time_source": cfg["service_time_source"],
            "service_time_us_per_token": service_time,
            "service_time_derivation": cfg.get("service_time_derivation"),
            "hbm_budget_source": cfg["hbm_budget_source"],
        },
        "offline_inputs_ready": offline_inputs_ready,
        "formal_ready": not blocking_gates,
        "blocking_gates": blocking_gates,
        "plan": plan.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "formal_ready": result["formal_ready"],
        "experts": len(experts),
        "predicted_p99_us": plan.predicted_p99_us,
        "predicted_cross_numa_bytes": plan.predicted_cross_numa_bytes,
        "migration_bytes": plan.migration_bytes,
        "max_gpu_memory_bytes": max(plan.gpu_memory_bytes.values()),
        "plan_sha256": plan.plan_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
