#!/usr/bin/env bash
# 作用：采集 GPU、NUMA、PCIe、CUDA 和 NCCL 硬件概况。
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env/project.env
source "$root/env/project.env"

run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)_hardware}"
out="${QTOPOMOE_ARTIFACTS}/hardware/${run_id}"
mkdir -p "$out"

capture() {
  local name=$1
  shift
  printf 'RUN %s\n' "$name"
  "$@" >"$out/$name.txt" 2>&1
  local rc=$?
  printf '%s\t%s\n' "$name" "$rc" >>"$out/status.tsv"
  return 0
}

printf 'item\trc\n' >"$out/status.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ >"$out/captured_at_utc.txt"
hostname >"$out/hostname.txt"

capture os-release cat /etc/os-release
capture uname uname -a
capture kernel-cmdline cat /proc/cmdline
capture virtualization bash -c 'systemd-detect-virt; rc=$?; [[ $rc -eq 0 || $rc -eq 1 ]]'
capture lscpu lscpu
capture lscpu-e lscpu -e
capture numa numactl --hardware
capture numastat numastat
capture memory free -h
capture mounts findmnt
capture lspci-tree lspci -tv
capture lspci-nvidia lspci -Dnnk -d 10de:
capture iommu-groups find /sys/kernel/iommu_groups -maxdepth 2 -type l
capture nvcc "$CUDACXX" --version
capture nvidia-smi-L nvidia-smi -L
capture nvidia-smi-q nvidia-smi -q
capture topo-m nvidia-smi topo -m
capture topo-p2p-read nvidia-smi topo -p2p r
capture topo-p2p-write nvidia-smi topo -p2p w
capture gpu-query nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.gpumax,pcie.link.gen.gpucurrent,pcie.link.gen.hostmax,pcie.link.width.max,pcie.link.width.current,memory.total,memory.used,memory.free,pstate,power.limit,temperature.gpu --format=csv
capture compute-processes nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
capture listening-ports ss -ltnp

python3 - "$out/status.tsv" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = []
for line in p.read_text().splitlines()[1:]:
    name, rc = line.split("\t")
    rows.append({"item": name, "rc": int(rc)})
summary = {
    "items": len(rows),
    "failed": [r for r in rows if r["rc"] != 0],
    "required_outputs": [
        "nvidia-smi-L.txt", "nvidia-smi-q.txt", "topo-m.txt",
        "topo-p2p-read.txt", "topo-p2p-write.txt", "numa.txt",
        "lspci-tree.txt", "gpu-query.txt"
    ],
}
(p.parent / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "$out"
