#!/usr/bin/env bash
# 作用：只读采集目标机器的 GPU、NUMA、P2P、NCCL 与软件环境证据。
set -Eeuo pipefail

# Read-only phase-0 collector for the 8xRTX 5090 Q-TopoMoE host.
# Usage:
#   bash q_topomoe_phase0_verify.sh [output_dir]
# Optional:
#   P2P_TEST_BIN=/path/to/p2pBandwidthLatencyTest
#   NCCL_TESTS_BIN_DIR=/path/to/nccl-tests/build
#   BENCH_PROFILE=quick|formal
#   TEST_P2P_DISABLED=0|1

OUT_DIR="${1:-artifacts/q_topomoe_phase0_$(date +%Y%m%d_%H%M%S)}"
BENCH_PROFILE="${BENCH_PROFILE:-quick}"
NCCL_TESTS_BIN_DIR="${NCCL_TESTS_BIN_DIR:-}"
P2P_TEST_BIN="${P2P_TEST_BIN:-}"
TEST_P2P_DISABLED="${TEST_P2P_DISABLED:-1}"

mkdir -p "$OUT_DIR"/{hardware,runtime,p2p,nccl}

capture() {
  local name="$1"
  shift
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
  } >"$OUT_DIR/$name.txt" 2>&1 || {
    local rc=$?
    printf 'FAILED rc=%s: %s\n' "$rc" "$name" >&2
    return 0
  }
}

capture hardware/date date --iso-8601=seconds
capture hardware/uname uname -a
capture hardware/os-release cat /etc/os-release
capture hardware/virtualization bash -lc 'systemd-detect-virt || true'
capture hardware/kernel-cmdline cat /proc/cmdline
capture hardware/lscpu lscpu
capture hardware/numa numactl --hardware
capture hardware/nvidia-smi-L nvidia-smi -L
capture hardware/nvidia-smi-q nvidia-smi -q
capture hardware/nvidia-smi-topo nvidia-smi topo -m
capture hardware/nvidia-smi-p2p-read nvidia-smi topo -p2p r
capture hardware/nvidia-smi-p2p-write nvidia-smi topo -p2p w
capture hardware/nvidia-smi-query nvidia-smi --query-gpu=index,uuid,pci.bus_id,pstate,memory.total,memory.free,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,power.limit,temperature.gpu --format=csv
capture hardware/lspci-tree lspci -tv
capture hardware/lspci-verbose lspci -nnvv
capture hardware/iommu-acs bash -lc "dmesg 2>/dev/null | grep -Ei 'DMAR|IOMMU|ACS' || true"
capture runtime/nvcc nvcc --version
capture runtime/ldconfig-nccl bash -lc 'ldconfig -p 2>/dev/null | grep -i nccl || true'
capture runtime/python-version python --version

if command -v python >/dev/null 2>&1; then
  python - <<'PY' >"$OUT_DIR/runtime/torch.json" 2>"$OUT_DIR/runtime/torch.stderr.txt" || true
import json
import platform

try:
    import torch
except Exception as exc:
    print(json.dumps({"import_error": repr(exc)}, indent=2))
    raise SystemExit(0)

result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "device_count": torch.cuda.device_count(),
    "cuda_available": torch.cuda.is_available(),
    "nccl_available": torch.distributed.is_nccl_available(),
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    "devices": [],
    "peer_access": [],
    "bf16_gemm": [],
}

try:
    result["nccl_version"] = list(torch.cuda.nccl.version())
except Exception as exc:
    result["nccl_version_error"] = repr(exc)

for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    result["devices"].append({
        "index": index,
        "name": prop.name,
        "capability": list(torch.cuda.get_device_capability(index)),
        "memory_bytes": prop.total_memory,
    })
    try:
        with torch.cuda.device(index):
            value = torch.randn((2048, 2048), device=f"cuda:{index}", dtype=torch.bfloat16)
            output = value @ value.T
            torch.cuda.synchronize(index)
            result["bf16_gemm"].append({"index": index, "ok": True, "sample": float(output[0, 0])})
    except Exception as exc:
        result["bf16_gemm"].append({"index": index, "ok": False, "error": repr(exc)})

peer_fn = getattr(torch.cuda, "can_device_access_peer", None)
for src in range(torch.cuda.device_count()):
    row = []
    for dst in range(torch.cuda.device_count()):
        if src == dst:
            row.append(True)
        elif peer_fn is None:
            row.append(None)
        else:
            row.append(bool(peer_fn(src, dst)))
    result["peer_access"].append(row)

print(json.dumps(result, indent=2))
PY
fi

if command -v torchrun >/dev/null 2>&1; then
  cat >"$OUT_DIR/runtime/nccl_smoke.py" <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
world = dist.get_world_size()

value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
expected_sum = world * (world + 1) / 2
assert value.item() == expected_sum, (rank, value.item(), expected_sum)

send = torch.arange(world, device=f"cuda:{rank}", dtype=torch.float32) + 100 * rank
recv = torch.empty_like(send)
dist.all_to_all_single(recv, send)
expected = torch.arange(world, device=f"cuda:{rank}", dtype=torch.float32) * 100 + rank
assert torch.equal(recv, expected), (rank, recv, expected)

dist.barrier()
if rank == 0:
    print(f"NCCL all_reduce + all_to_all PASS world={world}")
dist.destroy_process_group()
PY

  set +e
  NCCL_DEBUG=INFO \
  NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET \
  NCCL_TOPO_DUMP_FILE="$OUT_DIR/runtime/nccl-default-topology.xml" \
    timeout 180s torchrun --standalone --nproc-per-node=8 "$OUT_DIR/runtime/nccl_smoke.py" \
      >"$OUT_DIR/runtime/nccl-default.log" 2>&1
  NCCL_SMOKE_RC=$?
  set -e
  printf '%s\n' "$NCCL_SMOKE_RC" >"$OUT_DIR/runtime/nccl-default.rc"

  # Preserve the default failure first, then isolate IB/RDMA only as a diagnostic.
  if [[ "$NCCL_SMOKE_RC" -ne 0 ]]; then
    set +e
    NCCL_IB_DISABLE=1 \
    NCCL_NET=Socket \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET \
      timeout 180s torchrun --standalone --nproc-per-node=8 "$OUT_DIR/runtime/nccl_smoke.py" \
        >"$OUT_DIR/runtime/nccl-no-ib-socket.log" 2>&1
    NCCL_DIAG_RC=$?
    set -e
    printf '%s\n' "$NCCL_DIAG_RC" >"$OUT_DIR/runtime/nccl-no-ib-socket.rc"
  fi
else
  printf '%s\n' 'torchrun not found; NCCL smoke not run.' >"$OUT_DIR/runtime/NCCL_SMOKE_NOT_RUN.txt"
fi

if [[ -z "$P2P_TEST_BIN" ]]; then
  for candidate in \
    /usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest \
    /usr/local/cuda/samples/bin/x86_64/linux/release/p2pBandwidthLatencyTest; do
    if [[ -x "$candidate" ]]; then
      P2P_TEST_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$P2P_TEST_BIN" ]] && command -v p2pBandwidthLatencyTest >/dev/null 2>&1; then
  P2P_TEST_BIN="$(command -v p2pBandwidthLatencyTest)"
fi

if [[ -n "$P2P_TEST_BIN" && -x "$P2P_TEST_BIN" ]]; then
  capture p2p/p2pBandwidthLatencyTest "$P2P_TEST_BIN"
else
  printf '%s\n' 'p2pBandwidthLatencyTest not found; set P2P_TEST_BIN.' >"$OUT_DIR/p2p/NOT_RUN.txt"
fi

if [[ "$BENCH_PROFILE" == "formal" ]]; then
  MAX_BYTES="1G"
  WARMUPS="20"
  ITERS="100"
else
  MAX_BYTES="256M"
  WARMUPS="5"
  ITERS="20"
fi

if [[ -z "$NCCL_TESTS_BIN_DIR" ]]; then
  for candidate in /opt/nccl-tests/build /usr/local/nccl-tests/build; do
    if [[ -x "$candidate/all_reduce_perf" ]]; then
      NCCL_TESTS_BIN_DIR="$candidate"
      break
    fi
  done
fi

run_nccl() {
  local label="$1"
  local devices="$2"
  local binary="$3"
  local p2p_disabled="${4:-0}"
  local gpu_list=()
  IFS=',' read -r -a gpu_list <<<"$devices"
  local ngpus="${#gpu_list[@]}"
  local output="$OUT_DIR/nccl/${label}_${binary}_p2p${p2p_disabled}.txt"
  local topo_dump="$OUT_DIR/nccl/${label}_${binary}_p2p${p2p_disabled}_topology.xml"

  if [[ ! -x "$NCCL_TESTS_BIN_DIR/$binary" ]]; then
    printf 'SKIPPED: %s not found\n' "$NCCL_TESTS_BIN_DIR/$binary" >"$output"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$devices" \
  NCCL_DEBUG=INFO \
  NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET \
  NCCL_TOPO_DUMP_FILE="$topo_dump" \
  NCCL_P2P_DISABLE="$p2p_disabled" \
    timeout 600s "$NCCL_TESTS_BIN_DIR/$binary" \
      -b 8 -e "$MAX_BYTES" -f 2 -g "$ngpus" \
      -w "$WARMUPS" -n "$ITERS" -c 1 >"$output" 2>&1 || true
}

if [[ -n "$NCCL_TESTS_BIN_DIR" && -x "$NCCL_TESTS_BIN_DIR/all_reduce_perf" ]]; then
  printf '%s\n' "$NCCL_TESTS_BIN_DIR" >"$OUT_DIR/nccl/bin-dir.txt"

  # Representative pair paths: PIX, NODE, SYS.
  for spec in pix_01:0,1 node_02:0,2 sys_04:0,4; do
    label="${spec%%:*}"
    devices="${spec#*:}"
    run_nccl "$label" "$devices" sendrecv_perf 0
  done

  # MoE-relevant collectives on a NUMA-local group and all eight GPUs.
  for spec in numa0_0123:0,1,2,3 numa1_4567:4,5,6,7 all_01234567:0,1,2,3,4,5,6,7; do
    label="${spec%%:*}"
    devices="${spec#*:}"
    for binary in all_gather_perf reduce_scatter_perf all_reduce_perf alltoall_perf; do
      run_nccl "$label" "$devices" "$binary" 0
    done
  done

  # A/B only on representative groups; default NCCL may already avoid P2P.
  if [[ "$TEST_P2P_DISABLED" == "1" ]]; then
    run_nccl pix_01 0,1 sendrecv_perf 1
    run_nccl numa0_0123 0,1,2,3 all_gather_perf 1
    run_nccl all_01234567 0,1,2,3,4,5,6,7 all_gather_perf 1
    run_nccl all_01234567 0,1,2,3,4,5,6,7 reduce_scatter_perf 1
  fi
else
  printf '%s\n' 'nccl-tests not found; set NCCL_TESTS_BIN_DIR.' >"$OUT_DIR/nccl/NOT_RUN.txt"
fi

printf 'output_dir=%s\n' "$OUT_DIR"
printf 'profile=%s max_bytes=%s warmups=%s iters=%s\n' "$BENCH_PROFILE" "$MAX_BYTES" "$WARMUPS" "$ITERS"
printf '%s\n' 'Collection complete. Review every FAILED/SKIPPED/NOT_RUN marker before drawing conclusions.'
