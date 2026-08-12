#!/usr/bin/env bash
set -euo pipefail

# 在两组 GPU 上同时运行官方协议 A/B 分片。请用 nohup + setsid 启动本脚本，
# 这样 SSH 或上层客户端退出不会向服务进程传播 SIGHUP。

if [[ $# -ne 2 ]]; then
  echo "用法：$0 <nvfp4|fp8> <输出目录>" >&2
  exit 2
fi

FORMAT="$1"
OUT_ROOT="$2"
PROJECT_ROOT="${PROJECT_ROOT:-/data/moe}"
QUALITY_ROOT="${QUALITY_ROOT:-/data/models/test/qtopomoe_quality}"
VLLM_ENV="${VLLM_ENV:-/data/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129}"
PYTHON_BIN="$VLLM_ENV/bin/python"
VLLM_COMMAND=("$PYTHON_BIN" -m vllm.entrypoints.cli.main)
PORT_A="${PORT_A:-31620}"
PORT_B="${PORT_B:-31621}"
SEED="${SEED:-42}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

case "$FORMAT" in
  nvfp4)
    MODEL_PATH="${MODEL_PATH:-/data/models/test/redhatai_qwen36_nvfp4}"
    TP_SIZE=4
    GPU_A="0,1,2,3"
    GPU_B="4,5,6,7"
    EXTRA_ARGS=(--enable-expert-parallel)
    ;;
  fp8)
    MODEL_PATH="${MODEL_PATH:-/data/models/test/models/Qwen--Qwen3.6-35B-A3B-FP8/snapshots/master}"
    TP_SIZE=2
    GPU_A="0,1"
    GPU_B="4,5"
    EXTRA_ARGS=(--moe-backend triton)
    ;;
  *)
    echo "不支持的格式：$FORMAT" >&2
    exit 2
    ;;
esac

for required in "$PYTHON_BIN" "$PROJECT_ROOT/clients/quality_eval.py" \
  "$PROJECT_ROOT/serving/acceptance.sh" \
  "$QUALITY_ROOT/full_set_protocol_shard_a.jsonl" \
  "$QUALITY_ROOT/full_set_protocol_shard_b.jsonl"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少必需路径：$required" >&2
    exit 2
  fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "模型目录不存在：$MODEL_PATH" >&2
  exit 2
fi
if [[ -e "$OUT_ROOT" && "${RESUME:-0}" != "1" ]]; then
  echo "输出目录已存在；如需断点续跑，请设置 RESUME=1：$OUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/shard_a" "$OUT_ROOT/shard_b"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PATH="$VLLM_ENV/bin:$PATH"

SERVER_PID_A=""
SERVER_PID_B=""
CLIENT_PID_A=""
CLIENT_PID_B=""

write_status() {
  local state="$1"
  local detail="${2:-}"
  "$PYTHON_BIN" - "$OUT_ROOT/status.json" "$FORMAT" "$state" "$detail" \
    "$SERVER_PID_A" "$SERVER_PID_B" "$CLIENT_PID_A" "$CLIENT_PID_B" <<'PY'
import json
import os
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "qtopomoe.fullset_pair_status.v1",
    "format": sys.argv[2],
    "state": sys.argv[3],
    "detail": sys.argv[4],
    "updated_unix": time.time(),
    "manager_pid": os.getppid(),
    "server_pids": {"a": sys.argv[5] or None, "b": sys.argv[6] or None},
    "client_pids": {"a": sys.argv[7] or None, "b": sys.argv[8] or None},
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
temporary.replace(path)
PY
}

cleanup() {
  local pid
  for pid in "$CLIENT_PID_A" "$CLIENT_PID_B" "$SERVER_PID_A" "$SERVER_PID_B"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

launch_server() {
  local shard="$1"
  local gpu_ids="$2"
  local port="$3"
  local numa_node="$4"
  local served_name="qtopomoe-${FORMAT}-full-${shard}"
  local shard_dir="$OUT_ROOT/shard_${shard}"
  local command=(numactl "--cpunodebind=$numa_node" "--membind=$numa_node"
    "${VLLM_COMMAND[@]}" serve "$MODEL_PATH"
    --host 127.0.0.1 --port "$port" --served-model-name "$served_name"
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-prefix-caching --enable-prompt-tokens-details
    "${EXTRA_ARGS[@]}")
  printf '%q ' env CUDA_VISIBLE_DEVICES="$gpu_ids" "${command[@]}" > "$shard_dir/server.command.txt"
  printf '\n' >> "$shard_dir/server.command.txt"
  env CUDA_VISIBLE_DEVICES="$gpu_ids" "${command[@]}" > "$shard_dir/server.log" 2>&1 &
  echo $!
}

wait_ready() {
  local port="$1"
  local pid="$2"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

write_status "starting_servers"
SERVER_PID_A="$(launch_server a "$GPU_A" "$PORT_A" 0)"
SERVER_PID_B="$(launch_server b "$GPU_B" "$PORT_B" 1)"
write_status "waiting_for_servers"

if ! wait_ready "$PORT_A" "$SERVER_PID_A" || ! wait_ready "$PORT_B" "$SERVER_PID_B"; then
  write_status "server_start_failed" "请检查两个分片的 server.log"
  exit 1
fi

ROOT_URL="http://127.0.0.1:$PORT_A" SERVED_NAME="qtopomoe-${FORMAT}-full-a" \
  OUT_DIR="$OUT_ROOT/shard_a/acceptance" bash "$PROJECT_ROOT/serving/acceptance.sh"
ROOT_URL="http://127.0.0.1:$PORT_B" SERVED_NAME="qtopomoe-${FORMAT}-full-b" \
  OUT_DIR="$OUT_ROOT/shard_b/acceptance" bash "$PROJECT_ROOT/serving/acceptance.sh"

write_status "running_clients"
"$PYTHON_BIN" "$PROJECT_ROOT/clients/quality_eval.py" \
  --base-url "http://127.0.0.1:$PORT_A/v1" \
  --model "qtopomoe-${FORMAT}-full-a" \
  --manifest "$QUALITY_ROOT/full_set_protocol_shard_a.jsonl" \
  --output "$OUT_ROOT/shard_a/quality.results.jsonl" \
  --summary "$OUT_ROOT/shard_a/quality.summary.json" \
  --concurrency "$CONCURRENCY" --seed "$SEED" --timeout 1200 \
  --checkpoint-every 100 --resume > "$OUT_ROOT/shard_a/client.log" 2>&1 &
CLIENT_PID_A=$!
"$PYTHON_BIN" "$PROJECT_ROOT/clients/quality_eval.py" \
  --base-url "http://127.0.0.1:$PORT_B/v1" \
  --model "qtopomoe-${FORMAT}-full-b" \
  --manifest "$QUALITY_ROOT/full_set_protocol_shard_b.jsonl" \
  --output "$OUT_ROOT/shard_b/quality.results.jsonl" \
  --summary "$OUT_ROOT/shard_b/quality.summary.json" \
  --concurrency "$CONCURRENCY" --seed "$SEED" --timeout 1200 \
  --checkpoint-every 100 --resume > "$OUT_ROOT/shard_b/client.log" 2>&1 &
CLIENT_PID_B=$!
write_status "running_clients"

set +e
wait "$CLIENT_PID_A"
CLIENT_RC_A=$?
wait "$CLIENT_PID_B"
CLIENT_RC_B=$?
set -e

FINAL_STATE="$($PYTHON_BIN - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summaries = []
for shard in ("a", "b"):
    path = root / f"shard_{shard}" / "quality.summary.json"
    if not path.exists():
        print("client_failed")
        raise SystemExit
    summaries.append(json.loads(path.read_text()))
if any(item.get("failed", 0) for item in summaries):
    print("client_failed")
elif any(item.get("truncated", 0) for item in summaries):
    print("base_completed_rerun_required")
else:
    print("base_completed")
PY
)"
write_status "$FINAL_STATE" "client_rc_a=$CLIENT_RC_A client_rc_b=$CLIENT_RC_B"

if [[ "$FINAL_STATE" == "client_failed" ]]; then
  exit 1
fi
