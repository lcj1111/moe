#!/usr/bin/env python3
# 作用：根据权限受限的进程快照执行失败后服务恢复。
"""从权限受限的进程快照恢复服务，仅用于失败后的紧急回滚。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def health(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def gpu_pids(indices: list[int]) -> dict[int, list[int]]:
    result = {}
    for index in indices:
        completed = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        result[index] = [
            int(value) for value in re.findall(r"(?m)^\s*(\d+)\s*$", completed.stdout)
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    url = manifest["health_url"]
    expected_gpus = list(manifest["expected_gpu_ids"])
    if health(url) == 200:
        raise RuntimeError("目标健康端口已有服务，拒绝重复启动")
    deadline = time.time() + 900
    applications = {}
    while time.time() < deadline:
        applications = gpu_pids(expected_gpus)
        if not any(applications.values()):
            break
        time.sleep(5)
    if any(applications.values()):
        raise RuntimeError(f"目标GPU仍被占用: {applications}")
    target = manifest.get("stdout_target", "")
    if not isinstance(target, str) or not target.startswith("/"):
        target = str(args.status.with_suffix(".log"))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab", buffering=0) as output:
        process = subprocess.Popen(
            manifest["argv"], cwd=manifest["cwd"], env=manifest["environment"],
            stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"恢复服务提前退出: {process.returncode}")
        if health(url) == 200:
            break
        time.sleep(5)
    status = {
        "schema_version": "qtopomoe.emergency_service_restore.v1",
        "restored_unix": time.time(),
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "health_http": health(url),
        "gpu_ids": [index for index, pids in gpu_pids(list(range(8))).items() if pids],
    }
    args.status.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))
    return 0 if status["health_http"] == 200 and status["gpu_ids"] == expected_gpus else 1


if __name__ == "__main__":
    raise SystemExit(main())
