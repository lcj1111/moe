#!/usr/bin/env python3
"""等待既有服务空闲，执行有限 canary，并在任何退出路径恢复原服务。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CURRENT_CANARY: subprocess.Popen[Any] | None = None


def handle_signal(signum: int, _frame: Any) -> None:
    raise RuntimeError(f"canary守护进程收到信号{signum}")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, mode)
    temporary.replace(path)


def health_status(url: str, timeout: int = 5) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def listener_pid(port: int) -> int | None:
    output = subprocess.run(
        ["ss", "-ltnp"], check=True, capture_output=True, text=True
    ).stdout
    for line in output.splitlines():
        if re.search(rf":{port}\s", line):
            match = re.search(r"pid=(\d+)", line)
            if match:
                return int(match.group(1))
    return None


def proc_vector(path: Path) -> list[str]:
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in path.read_bytes().split(b"\0")
        if item
    ]


def gpu_pids(indices: list[int]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index in indices:
        completed = subprocess.run(
            [
                "nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi GPU{index}失败: {completed.stdout}")
        result[index] = [int(value) for value in re.findall(r"(?m)^\s*(\d+)\s*$", completed.stdout)]
    return result


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_group(pgid: int, timeout: int = 180) -> None:
    if not process_group_exists(pgid):
        return
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and process_group_exists(pgid):
        time.sleep(2)
    if process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.time() + 30
        while time.time() < deadline and process_group_exists(pgid):
            time.sleep(1)
    if process_group_exists(pgid):
        raise RuntimeError(f"进程组{pgid}未能停止")


def wait_gpus_idle(indices: list[int], timeout: int) -> None:
    deadline = time.time() + timeout
    last: dict[int, list[int]] = {}
    while time.time() < deadline:
        last = gpu_pids(indices)
        if not any(last.values()):
            return
        time.sleep(5)
    raise RuntimeError(f"GPU未在等待窗口内释放: {last}")


def metrics_load(url: str) -> tuple[float, float]:
    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode("utf-8", errors="replace")
    running = [
        float(value) for value in re.findall(
            r"(?m)^sglang:num_running_reqs\{[^\n]*\}\s+([0-9.eE+-]+)$", text
        )
    ]
    queued = [
        float(value) for value in re.findall(
            r"(?m)^sglang:num_queue_reqs\{[^\n]*\}\s+([0-9.eE+-]+)$", text
        )
    ]
    if not running or not queued:
        raise RuntimeError("无法从SGLang metrics读取运行/排队请求")
    return max(running), max(queued)


def wait_service_idle(plan: dict[str, Any], update) -> None:
    service = plan["既有服务"]
    metrics_url = service["health_url"].rsplit("/", 1)[0] + "/metrics"
    needed = int(service["idle_samples_required"])
    interval = int(service["idle_sample_interval_seconds"])
    deadline = time.time() + int(service["idle_wait_timeout_seconds"])
    consecutive = 0
    while time.time() < deadline:
        running, queued = metrics_load(metrics_url)
        consecutive = consecutive + 1 if running == 0 and queued == 0 else 0
        update(
            "waiting_existing_service_idle",
            running_requests=running,
            queued_requests=queued,
            consecutive_idle_samples=consecutive,
            required_idle_samples=needed,
        )
        if consecutive >= needed:
            return
        time.sleep(interval)
    raise RuntimeError("既有SGLang服务未在等待窗口内连续空闲")


def capture_service(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    service = plan["既有服务"]
    port = int(service["port"])
    pid = listener_pid(port)
    if pid is None:
        raise RuntimeError(f"端口{port}没有监听进程")
    proc = Path(f"/proc/{pid}")
    argv = proc_vector(proc / "cmdline")
    if service["expected_module"] not in " ".join(argv):
        raise RuntimeError("监听进程不是预期SGLang服务")
    if "--port" not in argv or int(argv[argv.index("--port") + 1]) != port:
        raise RuntimeError("SGLang端口参数不匹配")
    pgid = os.getpgid(pid)
    expected_gpus = list(service["expected_gpu_ids"])
    applications = gpu_pids(list(range(8)))
    if any(applications[index] for index in range(4)):
        raise RuntimeError(f"GPU0-3出现未知计算进程: {applications}")
    for index in expected_gpus:
        if not applications[index]:
            raise RuntimeError(f"GPU{index}未发现既有服务计算进程")
        foreign = [child for child in applications[index] if os.getpgid(child) != pgid]
        if foreign:
            raise RuntimeError(f"GPU{index}存在非目标进程组: {foreign}")
    environment = {}
    for entry in proc_vector(proc / "environ"):
        if "=" in entry:
            key, value = entry.split("=", 1)
            environment[key] = value
    manifest = {
        "schema_version": "qtopomoe.canary_service_restore_manifest.v1",
        "captured_unix": time.time(),
        "original_pid": pid,
        "original_pgid": pgid,
        "cwd": os.readlink(proc / "cwd"),
        "stdout_target": os.readlink(proc / "fd/1"),
        "argv": argv,
        "environment": environment,
        "health_url": service["health_url"],
        "expected_gpu_ids": expected_gpus,
    }
    atomic_json(root / "private_service_restore_manifest.json", manifest, mode=0o600)
    return manifest


def start_canary(plan: dict[str, Any], repo: Path, root: Path) -> subprocess.Popen[Any]:
    runtime = plan["运行环境"]
    python_bin = runtime["python_bin"]
    command = [
        python_bin, "-m", "vllm.entrypoints.cli.main", "serve", runtime["model"],
        "--host", runtime["host"], "--port", str(runtime["port"]),
        "--served-model-name", runtime["served_model_name"],
        "--tensor-parallel-size", str(runtime["tensor_parallel_size"]),
        "--data-parallel-size", "1",
        "--max-model-len", str(runtime["max_model_len"]),
        "--max-num-seqs", str(runtime["max_num_seqs"]),
        "--gpu-memory-utilization", str(runtime["gpu_memory_utilization"]),
        "--enable-prompt-tokens-details", "--enable-expert-parallel", "--enable-eplb",
        "--eplb-config",
        json.dumps({
            "window_size": runtime["eplb_window_size"],
            "step_interval": runtime["eplb_step_interval"],
            "num_redundant_experts": 0,
            "log_balancedness": True,
            "log_balancedness_interval": 8,
            "use_async": False,
            "communicator": "torch_nccl",
        }, separators=(",", ":")),
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in runtime["gpu_ids"]),
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "QTOPOMOE_EPLB_PLAN": runtime["runtime_plan"],
        "QTOPOMOE_EPLB_DEFER_CALLS": str(runtime["defer_calls"]),
        "PYTHONPATH": str(repo / "runtime_patches" / "qtopomoe_eplb"),
    })
    log = (root / "server.log").open("ab", buffering=0)
    process = subprocess.Popen(
        command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    atomic_json(root / "canary_service_process.json", {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "port": runtime["port"],
        "started_unix": time.time(),
        "command": command,
    })
    return process


def wait_canary_health(process: subprocess.Popen[Any], plan: dict[str, Any]) -> None:
    runtime = plan["运行环境"]
    url = f"http://{runtime['host']}:{runtime['port']}/v1/models"
    deadline = time.time() + 1800
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"canary服务启动时退出: {process.returncode}")
        if health_status(url) == 200:
            return
        time.sleep(5)
    raise RuntimeError("canary服务健康检查超时")


def run_phase(plan: dict[str, Any], repo: Path, root: Path, phase: str, offset: int) -> None:
    runtime = plan["运行环境"]
    workload = plan["负载"]
    phase_dir = root / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    command = [
        runtime["python_bin"], str(repo / "clients" / "smoke.py"),
        "--base-url", f"http://{runtime['host']}:{runtime['port']}/v1",
        "--model", runtime["served_model_name"],
        "--tokenizer", runtime["model"],
        "--input-tokens", str(workload["input_tokens"]),
        "--output-tokens", str(workload["output_tokens"]),
        "--concurrency", str(workload["concurrency"]),
        "--requests", str(workload["requests_per_phase"]),
        "--seed", str(int(workload["seed"]) + offset),
        "--stream-seed", str(int(workload["seed"]) + offset),
        "--prefix-cache-pct", str(workload["prefix_cache_pct"]),
        "--arrival-mode", workload["arrival_mode"],
        "--sample-service-metrics", "--require-arrival-gate",
        "--observation-phase", "post_workload",
        "--timeout", str(workload["request_timeout_seconds"]),
        "--output", str(phase_dir / "requests.jsonl"),
        "--summary", str(phase_dir / "summary.json"),
    ]
    with (phase_dir / "client.log").open("wb") as log:
        completed = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"{phase}客户端失败: {completed.returncode}")
    summary = load(phase_dir / "summary.json")
    if summary.get("failed") != 0 or summary.get("completed") != summary.get("requests"):
        raise RuntimeError(f"{phase}请求Gate失败")


def restore_service(manifest: dict[str, Any], plan: dict[str, Any], root: Path) -> dict[str, Any]:
    service = plan["既有服务"]
    if health_status(service["health_url"]) == 200:
        raise RuntimeError("恢复前端口已被未知健康服务占用")
    wait_gpus_idle(list(service["expected_gpu_ids"]), 900)
    target = manifest.get("stdout_target", "")
    if not isinstance(target, str) or not target.startswith("/"):
        target = str(root / "restored_sglang.log")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab", buffering=0) as output:
        process = subprocess.Popen(
            manifest["argv"], cwd=manifest["cwd"], env=manifest["environment"],
            stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + int(service["restore_timeout_seconds"])
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang恢复进程提前退出: {process.returncode}")
        if health_status(service["health_url"]) == 200:
            break
        time.sleep(5)
    health = health_status(service["health_url"])
    current = listener_pid(int(service["port"]))
    current_argv = proc_vector(Path(f"/proc/{current}/cmdline")) if current else []
    applications = gpu_pids(list(range(8)))
    active_gpus = [index for index, pids in applications.items() if pids]
    status = {
        "schema_version": "qtopomoe.canary_rollback_status.v1",
        "restored_unix": time.time(),
        "pid": current,
        "pgid": os.getpgid(current) if current else None,
        "health_http": health,
        "process_alive": current is not None,
        "command_matches": current_argv == manifest["argv"],
        "gpu_ids": active_gpus,
    }
    atomic_json(root / "rollback_status.json", status)
    return status


def main() -> int:
    global CURRENT_CANARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = load(args.plan)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "status.json"
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def update(stage: str, **extra: Any) -> None:
        atomic_json(status_path, {
            "schema_version": "qtopomoe.phase8_selector_limited_canary_status.v1",
            "stage": stage,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            **extra,
        })

    policy = repo / plan["准入依据"]["policy"]
    evidence = repo / plan["准入依据"]["independent_gate"]
    runtime_plan = Path(plan["运行环境"]["runtime_plan"])
    checks = {
        "policy": policy.exists() and sha256(policy) == plan["准入依据"]["policy_sha256"],
        "independent_gate": (
            evidence.exists()
            and sha256(evidence) == plan["准入依据"]["independent_gate_sha256"]
        ),
        "runtime_plan": (
            runtime_plan.exists()
            and sha256(runtime_plan) == plan["运行环境"]["runtime_plan_file_sha256"]
        ),
        "python": Path(plan["运行环境"]["python_bin"]).exists(),
        "model": Path(plan["运行环境"]["model"]).exists(),
    }
    if not all(checks.values()):
        update("preflight_failed", checks=checks)
        raise RuntimeError(f"冻结输入预检失败: {checks}")
    atomic_json(root / "canary_plan.snapshot.json", plan)
    update("waiting_existing_service_idle", checks=checks)
    manifest: dict[str, Any] | None = None
    pipeline_error: str | None = None
    restore_error: str | None = None
    try:
        wait_service_idle(plan, update)
        manifest = capture_service(plan, root)
        update("stopping_existing_service", original_pid=manifest["original_pid"])
        stop_group(int(manifest["original_pgid"]))
        wait_gpus_idle(list(range(8)), 600)
        update("launching_canary_service")
        CURRENT_CANARY = start_canary(plan, repo, root)
        wait_canary_health(CURRENT_CANARY, plan)
        update("running_stable")
        run_phase(plan, repo, root, "stable", 0)
        if "QTOPOMOE_EPLB_PLAN_APPLIED" in (root / "server.log").read_text(
            encoding="utf-8", errors="replace"
        ):
            raise RuntimeError("stable阶段结束前计划已提前应用")
        update("running_migration")
        run_phase(plan, repo, root, "migration", 1)
        if "QTOPOMOE_EPLB_PLAN_APPLIED" not in (root / "server.log").read_text(
            encoding="utf-8", errors="replace"
        ):
            raise RuntimeError("migration阶段未观察到计划应用")
        update("running_recovery")
        run_phase(plan, repo, root, "recovery", 2)
        update("canary_workload_completed")
    except BaseException as error:
        pipeline_error = repr(error)
        update("canary_pipeline_failed", error=pipeline_error)
    finally:
        try:
            if CURRENT_CANARY is not None:
                stop_group(os.getpgid(CURRENT_CANARY.pid))
                CURRENT_CANARY = None
            if manifest is not None:
                update("restoring_existing_service", pipeline_error=pipeline_error)
                rollback = restore_service(manifest, plan, root)
                update("existing_service_restored", rollback=rollback, pipeline_error=pipeline_error)
        except BaseException as error:
            restore_error = repr(error)
            update("restore_failed", pipeline_error=pipeline_error, restore_error=restore_error)
    if restore_error:
        return 2
    if pipeline_error:
        return 1
    analyzer = repo / "scripts" / "analyze_phase8_selector_limited_canary.py"
    command = [
        plan["运行环境"]["python_bin"], str(analyzer),
        "--root", str(root), "--server-log", str(root / "server.log"),
        "--runtime-plan", str(runtime_plan), "--canary-plan", str(args.plan),
        "--rollback-status", str(root / "rollback_status.json"),
        "--output", str(root / "canary_gate.json"),
    ]
    completed = subprocess.run(command, cwd=repo)
    gate = load(root / "canary_gate.json")
    update("completed", gate_status=gate["status"], analyzer_returncode=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
