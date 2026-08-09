#!/usr/bin/env python3
"""Run seeded, resumable Phase 8 service repetitions on an idle GPU host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_schedule(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"candidate_id": candidate["candidate_id"], "repeat": repeat}
        for repeat in range(1, int(plan["repeats_per_candidate"]) + 1)
        for candidate in plan["candidates"]
    ]
    random.Random(int(plan["seed"])).shuffle(rows)
    return [{"order": index + 1, **row} for index, row in enumerate(rows)]


def verify_runtime(plan: dict[str, Any], vllm_bin: str,
                   python_bin: str) -> dict[str, str]:
    for path in (Path(vllm_bin), Path(python_bin)):
        if not path.exists():
            raise RuntimeError(f"missing frozen runtime executable: {path}")
    code = (
        "import json,sys,vllm; "
        "print(json.dumps({'python':sys.executable,'vllm_version':vllm.__version__,"
        "'vllm_file':vllm.__file__}))")
    runtime = json.loads(subprocess.run(
        [python_bin, "-c", code], check=True, capture_output=True,
        text=True).stdout.strip())
    expected = plan["software"]["expected_vllm_version"]
    if runtime["vllm_version"] != expected:
        raise RuntimeError(
            f"frozen vLLM Gate failed: actual={runtime['vllm_version']} expected={expected}")
    runtime.update({"vllm_bin": str(Path(vllm_bin).resolve()),
                    "vllm_bin_sha256": sha256(Path(vllm_bin))})
    return runtime


def busy_gpu_ids() -> set[int]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    uuid_to_index = {
        uuid.strip(): int(index.strip())
        for index, uuid in (row.split(",", 1) for row in gpu_rows)
    }
    app_rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    return {uuid_to_index[row.strip()] for row in app_rows if row.strip() in uuid_to_index}


def health_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def stop_group(process: subprocess.Popen[Any], timeout: int = 60) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_client(python_bin: str, repo: Path, base_url: str, served_model: str,
               tokenizer: str, cell: dict[str, Any], seed: int,
               output: Path, summary: Path, timeout: int) -> None:
    command = [
        python_bin, str(repo / "clients" / "smoke.py"),
        "--base-url", base_url, "--model", served_model,
        "--tokenizer", tokenizer,
        "--input-tokens", str(cell["input_tokens"]),
        "--output-tokens", str(cell["output_tokens"]),
        "--concurrency", str(cell["concurrency"]),
        "--requests", str(cell["requests"]), "--seed", str(seed),
        "--timeout", str(timeout), "--output", str(output),
        "--summary", str(summary),
    ]
    subprocess.run(command, cwd=repo, check=True)


def audit_summary(path: Path, cell: dict[str, Any]) -> None:
    summary = load_json(path)
    checks = {
        "failed": summary.get("failed") == 0,
        "completed": summary.get("completed") == summary.get("requests"),
        "input_tokens": summary.get("input_tokens_actual") == cell["input_tokens"],
        "concurrency": summary.get("concurrency") == cell["concurrency"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"summary Gate failed for {path}: {checks}")


def service_command(plan: dict[str, Any], candidate: dict[str, Any],
                    vllm_bin: str, served_model: str) -> list[str]:
    command = []
    if candidate.get("numa_args"):
        command.extend(["numactl", *candidate["numa_args"]])
    command.extend([
        vllm_bin, "serve", candidate["model_path"],
        "--host", plan["host"], "--port", str(plan["port"]),
        "--served-model-name", served_model,
        "--tensor-parallel-size", str(candidate["tp"]),
        "--data-parallel-size", str(candidate["dp"]),
        "--max-model-len", str(plan["max_model_len"]),
        "--max-num-seqs", str(plan["max_num_seqs"]),
        "--gpu-memory-utilization", str(plan["gpu_memory_utilization"]),
    ])
    if candidate["enable_expert_parallel"]:
        command.append("--enable-expert-parallel")
    if candidate.get("moe_backend"):
        command.extend(["--moe-backend", candidate["moe_backend"]])
    return command


def run_one(plan: dict[str, Any], candidate: dict[str, Any], repeat: int,
            matrix: dict[str, Any], repo: Path, root: Path,
            vllm_bin: str, python_bin: str, runtime: dict[str, str]) -> None:
    run_id = f"{candidate['candidate_id']}__r{repeat:02d}"
    run_dir = root / "runs" / run_id
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("status") == "workload_passed":
            print(json.dumps({"run_id": run_id, "status": "resume_skip"}))
            return
        raise RuntimeError(f"refusing to overwrite incomplete run {run_dir}")
    selected = set(int(value) for value in candidate["gpu_ids"])
    busy = busy_gpu_ids() & selected
    if busy:
        raise RuntimeError(f"selected GPUs are busy before {run_id}: {sorted(busy)}")

    run_dir.mkdir(parents=True)
    served_model = "qtopomoe-" + run_id.replace("__", "-")
    command = service_command(plan, candidate, vllm_bin, served_model)
    meta = {
        "schema_version": "qtopomoe.phase8_repeated_run.v1",
        "run_id": run_id, "candidate_id": candidate["candidate_id"],
        "repeat": repeat, "status": "launching", "model_path": candidate["model_path"],
        "gpu_ids": candidate["gpu_ids"], "tp": candidate["tp"], "dp": candidate["dp"],
        "ep_enabled": candidate["enable_expert_parallel"],
        "expected_ep_ranks": (candidate["tp"] * candidate["dp"]
                              if candidate["enable_expert_parallel"] else 0),
        "actual_ep_ranks": 0, "moe_backend": candidate.get("moe_backend"),
        "backend_log_pattern": candidate["backend_log_pattern"],
        "numa_args": candidate.get("numa_args", []), "command": command,
        "runtime": runtime,
        "started_unix": time.time(),
    }
    write_json(meta_path, meta)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in candidate["gpu_ids"]),
        "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
    server_log = (run_dir / "server.log").open("w", encoding="utf-8")
    gpu_log = (run_dir / "gpu_memory.csv").open("w", encoding="utf-8")
    gpu_log.write("timestamp,index,memory.used,utilization.gpu,power.draw\n")
    gpu_log.flush()
    server = subprocess.Popen(command, cwd=repo, env=environment, stdout=server_log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits", "--loop-ms=500"],
        stdout=gpu_log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        health_url = f"http://{plan['host']}:{plan['port']}/v1/models"
        deadline = time.time() + int(plan["health_timeout_seconds"])
        while time.time() < deadline and not health_ready(health_url):
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode} during startup")
            time.sleep(2)
        if not health_ready(health_url):
            raise RuntimeError("server health timeout")
        base_url = f"http://{plan['host']}:{plan['port']}/v1"
        warmup = {"input_tokens": 256, "output_tokens": 64,
                  "concurrency": 8, "requests": 32}
        run_client(python_bin, repo, base_url, served_model, candidate["model_path"],
                   warmup, int(plan["seed"]), run_dir / "warmup.jsonl",
                   run_dir / "warmup.summary.json", int(plan["request_timeout_seconds"]))
        audit_summary(run_dir / "warmup.summary.json", warmup)
        server_log.flush()
        log_text = (run_dir / "server.log").read_text(encoding="utf-8", errors="replace")
        if candidate["backend_log_pattern"] not in log_text:
            raise RuntimeError("backend log Gate failed")
        ep_denominators = [int(value) for value in re.findall(r"EP Rank \d+/(\d+)", log_text)]
        actual_ep = max(ep_denominators, default=0)
        if actual_ep != meta["expected_ep_ranks"]:
            raise RuntimeError(
                f"EP rank Gate failed: actual={actual_ep} expected={meta['expected_ep_ranks']}")
        meta.update({"status": "warmup_passed", "actual_ep_ranks": actual_ep,
                     "warmup_summary_sha256": sha256(run_dir / "warmup.summary.json")})
        write_json(meta_path, meta)
        for index, cell in enumerate(matrix["cells"]):
            cell_dir = run_dir / "workloads" / cell["id"]
            cell_dir.mkdir(parents=True)
            cell_seed = int(plan["seed"]) + repeat * 100 + index
            summary_path = cell_dir / "summary.json"
            run_client(python_bin, repo, base_url, served_model, candidate["model_path"],
                       cell, cell_seed, cell_dir / "requests.jsonl", summary_path,
                       int(plan["request_timeout_seconds"]))
            audit_summary(summary_path, cell)
        meta.update({"status": "workload_passed", "completed_unix": time.time(),
                     "completed_cells": len(matrix["cells"])})
        write_json(meta_path, meta)
        print(json.dumps({"run_id": run_id, "status": meta["status"]}))
    except BaseException as error:
        meta.update({"status": "failed", "failed_unix": time.time(),
                     "failure": repr(error)})
        write_json(meta_path, meta)
        raise
    finally:
        stop_group(server)
        stop_group(monitor, timeout=10)
        server_log.close()
        gpu_log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vllm-bin")
    parser.add_argument("--python-bin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load_json(args.plan)
    vllm_bin = args.vllm_bin or plan["software"]["vllm_bin"]
    python_bin = args.python_bin or plan["software"]["python_bin"]
    runtime = verify_runtime(plan, vllm_bin, python_bin)
    matrix_path = repo / plan["workload_matrix"]
    matrix = load_json(matrix_path)
    candidates = {row["candidate_id"]: row for row in plan["candidates"]}
    schedule = build_schedule(plan)
    manifest = {
        "schema_version": "qtopomoe.phase8_repeated_schedule.v1",
        "plan": str(args.plan.resolve()), "plan_sha256": sha256(args.plan),
        "workload_matrix": str(matrix_path), "workload_matrix_sha256": sha256(matrix_path),
        "seed": plan["seed"], "runtime": runtime, "schedule": schedule,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_root / "schedule.json"
    if schedule_path.exists() and load_json(schedule_path) != manifest:
        raise RuntimeError("existing schedule does not match frozen plan")
    write_json(schedule_path, manifest)
    status_path = args.output_root / "status.json"
    for item in schedule:
        write_json(status_path, {"status": "running", "current": item,
                                 "updated_unix": time.time()})
        run_one(plan, candidates[item["candidate_id"]], item["repeat"], matrix,
                repo, args.output_root, vllm_bin, python_bin, runtime)
        write_json(status_path, {"status": "cooldown", "completed": item,
                                 "updated_unix": time.time()})
        time.sleep(int(plan["cooldown_seconds"]))
    write_json(status_path, {"status": "completed", "completed_runs": len(schedule),
                             "updated_unix": time.time()})


if __name__ == "__main__":
    main()
