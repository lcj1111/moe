#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/project.env"
FORMAT="${FORMAT:-fp8}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT_BASE="${PORT_BASE:-31100}"
TOPOLOGIES="${TOPOLOGIES:-tp2_node_0_2,tp2_node_1_3,tp2_node_4_6,tp2_node_5_7,tp2_pix_0_1,tp2_pix_2_3,tp2_pix_4_5,tp2_pix_6_7,tp4_numa0,tp4_numa1,tp8_sys}"
SEED="${SEED:-42}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"

case "$FORMAT" in
  fp8) MODEL_PATH="$QTOPOMOE_FP8_MODEL"; SERVED_NAME="qwen3.6-35b-a3b-fp8"; SERVE_ENV="$QTOPOMOE_SGLANG_VENV" ;;
  bf16) MODEL_PATH="$QTOPOMOE_BF16_MODEL"; SERVED_NAME="qwen3.6-35b-a3b-bf16"; SERVE_ENV="$QTOPOMOE_SGLANG_VENV" ;;
  *) echo "unsupported FORMAT=$FORMAT" >&2; exit 2 ;;
esac

RUN_ROOT="$QTOPOMOE_ARTIFACTS/phase1/$RUN_ID/$FORMAT"
mkdir -p "$RUN_ROOT"
MANIFEST="$RUN_ROOT/matrix.tsv"
printf 'format\ttopology\tgpu_ids\ttp_size\tnuma_bind\tstatus\trun_dir\n' > "$MANIFEST"

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
  python3 - "$RUN_ROOT" "$FORMAT" "$MODEL_PATH" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]) / "format_status.json"
out.write_text(json.dumps({"format": sys.argv[2], "status": "blocked", "reason": "model_path_missing", "model_path": sys.argv[3]}, indent=2) + "\n")
print(out)
PY
  exit 0
fi

gpu_ids=""; tp_size=""; numa_bind=""
topology_args() {
  case "$1" in
    tp2_node_0_2) gpu_ids=0,2; tp_size=2; numa_bind=node0 ;;
    tp2_node_1_3) gpu_ids=1,3; tp_size=2; numa_bind=node0 ;;
    tp2_node_4_6) gpu_ids=4,6; tp_size=2; numa_bind=node1 ;;
    tp2_node_5_7) gpu_ids=5,7; tp_size=2; numa_bind=node1 ;;
    tp2_pix_0_1) gpu_ids=0,1; tp_size=2; numa_bind=node0 ;;
    tp2_pix_2_3) gpu_ids=2,3; tp_size=2; numa_bind=node0 ;;
    tp2_pix_4_5) gpu_ids=4,5; tp_size=2; numa_bind=node1 ;;
    tp2_pix_6_7) gpu_ids=6,7; tp_size=2; numa_bind=node1 ;;
    tp4_numa0) gpu_ids=0,1,2,3; tp_size=4; numa_bind=node0 ;;
    tp4_numa1) gpu_ids=4,5,6,7; tp_size=4; numa_bind=node1 ;;
    tp8_sys) gpu_ids=0,1,2,3,4,5,6,7; tp_size=8; numa_bind=interleave ;;
    *) return 1 ;;
  esac
}
wait_health() {
  local root_url="$1" pid="$2"
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 5 "$root_url/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
    sleep 5
  done
  return 1
}

IFS=',' read -r -a requested <<< "$TOPOLOGIES"
for index in "${!requested[@]}"; do
  topology="${requested[$index]}"
  if ! topology_args "$topology"; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$FORMAT" "$topology" "" "" "" "invalid_topology" "" >> "$MANIFEST"
    continue
  fi
  port=$((PORT_BASE + index)); run_dir="$RUN_ROOT/$topology"; mkdir -p "$run_dir/server" "$run_dir/client" "$run_dir/metrics"
  root_url="http://127.0.0.1:$port"
  if ss -ltn | grep -q ":$port "; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$FORMAT" "$topology" "$gpu_ids" "$tp_size" "$numa_bind" "port_busy" "$run_dir" >> "$MANIFEST"
    continue
  fi
  case "$numa_bind" in
    node0) ncmd=(numactl --cpunodebind=0 --membind=0) ;;
    node1) ncmd=(numactl --cpunodebind=1 --membind=1) ;;
    interleave) ncmd=(numactl --interleave=0,1) ;;
  esac
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader > "$run_dir/server/gpu_before.csv" 2>&1 || true
  "${ncmd[@]}" env MODEL_PATH="$MODEL_PATH" SERVED_NAME="$SERVED_NAME" GPU_IDS="$gpu_ids" TP_SIZE="$tp_size" PORT="$port" SERVE_ENV="$SERVE_ENV" BACKEND=sglang MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC" NCCL_IB_DISABLE=1 nohup "$ROOT/serving/start_server.sh" > "$run_dir/server/server.log" 2>&1 < /dev/null &
  pid=$!; echo "$pid" > "$run_dir/server/server.pid"
  status="startup_failed"
  if wait_health "$root_url" "$pid"; then
    if ROOT_URL="$root_url" SERVED_NAME="$SERVED_NAME" OUT_DIR="$run_dir/client/acceptance" "$ROOT/serving/acceptance.sh" > "$run_dir/client/acceptance.stdout" 2>&1; then
      status="accepted"
      for workload in short medium; do
        if [[ "$workload" == short ]]; then input_tokens=256; output_tokens=128; else input_tokens=2048; output_tokens=256; fi
        for concurrency in 1 8 32; do
          output="$run_dir/client/${workload}_c${concurrency}.jsonl"; summary="$run_dir/client/${workload}_c${concurrency}.summary.json"
          if ! "$SERVE_ENV/bin/python" "$ROOT/clients/smoke.py" --base-url "$root_url/v1" --model "$SERVED_NAME" --input-tokens "$input_tokens" --output-tokens "$output_tokens" --concurrency "$concurrency" --requests "$concurrency" --seed "$SEED" --output "$output" --summary "$summary" > "$summary.stdout" 2>&1; then status="smoke_failed"; fi
        done
      done
    else
      status="acceptance_failed"
    fi
  else
    tail -80 "$run_dir/server/server.log" > "$run_dir/server/startup_tail.log" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader > "$run_dir/server/gpu_after.csv" 2>&1 || true
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader > "$run_dir/server/gpu_released.csv" 2>&1 || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$FORMAT" "$topology" "$gpu_ids" "$tp_size" "$numa_bind" "$status" "$run_dir" >> "$MANIFEST"
done

python3 - "$RUN_ROOT" "$MANIFEST" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); manifest = pathlib.Path(sys.argv[2]); rows = []
for line in manifest.read_text().splitlines()[1:]:
    if line.strip():
        f, t, g, tp, n, status, run = line.split("\t")
        rows.append({"format": f, "topology": t, "gpu_ids": g, "tp_size": int(tp) if tp else None, "numa_bind": n, "status": status, "run_dir": run})
(root / "phase1_summary.json").write_text(json.dumps({"run_root": str(root), "rows": rows}, indent=2) + "\n")
print(json.dumps({"run_root": str(root), "rows": len(rows), "statuses": {s: sum(1 for x in rows if x["status"] == s) for s in sorted({x["status"] for x in rows})}}, ensure_ascii=False))
PY
