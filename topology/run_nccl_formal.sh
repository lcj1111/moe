#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env/project.env
source "$root/env/project.env"

run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)_nccl_formal}"
out="${QTOPOMOE_ARTIFACTS}/raw/${run_id}/nccl"
mkdir -p "$out/logs"

bin="$QTOPOMOE_NCCL_TESTS"
nccl_lib="$QTOPOMOE_SGLANG_VENV/lib/python3.12/site-packages/nvidia/nccl/lib"
cu13_lib="$QTOPOMOE_SGLANG_VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${nccl_lib}:${cu13_lib}:${CUDA_HOME}/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID

summary="$out/run-status.tsv"
printf 'collective\tlabel\trepetition\tdevices\tnuma_policy\trc\n' >"$summary"

monitor="$out/pcie-power-monitor.csv"
(
  printf 'timestamp,index,pstate,pcie_gen,pcie_width,power_w,temp_c,util_gpu,memory_used_mib\n'
  while true; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    nvidia-smi --query-gpu=index,pstate,pcie.link.gen.current,pcie.link.width.current,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | sed "s/^/${timestamp},/"
    sleep 1
  done
) >"$monitor" 2>&1 &
monitor_pid=$!

cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

run_case() {
  local collective=$1 label=$2 repetition=$3 devices=$4 policy=$5 ngpu=$6 max_size=$7
  local executable="$bin/${collective}_perf"
  local log="$out/logs/${collective}_${label}_r${repetition}.log"
  local -a numa_args
  case "$policy" in
    node0) numa_args=(numactl --cpunodebind=0 --membind=0) ;;
    node1) numa_args=(numactl --cpunodebind=1 --membind=1) ;;
    interleave) numa_args=(numactl --interleave=0,1) ;;
    *) numa_args=() ;;
  esac

  if [[ ! -x "$executable" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t127\n' \
      "$collective" "$label" "$repetition" "$devices" "$policy" >>"$summary"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$devices" "${numa_args[@]}" \
    "$executable" -b 4K -e "$max_size" -f 4 -g "$ngpu" -w 20 -n 100 -c 1 \
    >"$log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$collective" "$label" "$repetition" "$devices" "$policy" "$rc" >>"$summary"
}

for repetition in 1 2 3 4 5; do
  # Representative two-GPU PIX, NODE, and SYS paths.
  run_case all_reduce pix01 "$repetition" 0,1 node0 2 256M
  run_case all_reduce node02 "$repetition" 0,2 node0 2 256M
  run_case all_reduce sys04_bind0 "$repetition" 0,4 node0 2 256M
  run_case sendrecv pix01 "$repetition" 0,1 node0 2 256M
  run_case sendrecv node02 "$repetition" 0,2 node0 2 256M
  run_case sendrecv sys04_bind0 "$repetition" 0,4 node0 2 256M

  # Local NUMA domains and full-system collectives.
  for collective in all_gather reduce_scatter; do
    run_case "$collective" numa0 "$repetition" 0,1,2,3 node0 4 256M
    run_case "$collective" numa1 "$repetition" 4,5,6,7 node1 4 256M
    run_case "$collective" all8 "$repetition" 0,1,2,3,4,5,6,7 interleave 8 256M
  done
  run_case alltoall numa0 "$repetition" 0,1,2,3 node0 4 64M
  run_case alltoall numa1 "$repetition" 4,5,6,7 node1 4 64M
  run_case alltoall all8 "$repetition" 0,1,2,3,4,5,6,7 interleave 8 64M
done

cleanup
trap - EXIT

cat "$summary"
if awk -F '\t' 'NR > 1 && $6 != 0 {bad=1} END {exit bad}' "$summary"; then
  echo "NCCL_FORMAL_ALL_COMMANDS_PASS"
else
  echo "NCCL_FORMAL_COMMAND_FAILURE" >&2
  exit 1
fi
