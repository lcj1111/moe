#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/qwen35_cleanroom.env"

BACKEND="${BACKEND:?BACKEND must be vllm or sglang}"
MODEL_PATH="${MODEL_PATH:-$QTOPOMOE_W4A16_SOURCE}"
GPU_IDS="${GPU_IDS:-0}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-31310}"
SERVED_NAME="${SERVED_NAME:-qtopomoe-w4a16-gate}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
SMOKE_REQUESTS="${SMOKE_REQUESTS:-0}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-8}"
SMOKE_INPUT_TOKENS="${SMOKE_INPUT_TOKENS:-256}"
SMOKE_OUTPUT_TOKENS="${SMOKE_OUTPUT_TOKENS:-64}"
SMOKE_SEED="${SMOKE_SEED:-42}"
QUALITY_MANIFEST="${QUALITY_MANIFEST:-}"
QUALITY_CONCURRENCY="${QUALITY_CONCURRENCY:-8}"
QUALITY_TIMEOUT="${QUALITY_TIMEOUT:-3600}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-auto}"

case "$BACKEND" in
  vllm)
    SERVE_ENV="$QTOPOMOE_CLEANROOM_ROOT/venvs/vllm-$QTOPOMOE_VLLM_COMMIT"
    ;;
  sglang)
    SERVE_ENV="$QTOPOMOE_CLEANROOM_ROOT/venvs/sglang-$QTOPOMOE_SGLANG_COMMIT"
    ;;
  *)
    echo "Unsupported BACKEND=$BACKEND" >&2
    exit 2
    ;;
esac

test -d "$MODEL_PATH"
test -x "$SERVE_ENV/bin/python"
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "Port $PORT is already occupied" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
SERVER_LOG="$OUT_DIR/server.log"
PID_FILE="$OUT_DIR/server.pid"
STATUS_FILE="$OUT_DIR/gate_status.json"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PATH="$SERVE_ENV/bin:$PATH"
unset PYTHONPATH

nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader > "$OUT_DIR/gpu_before.csv"
find "$MODEL_PATH" -maxdepth 1 -type f \
  \( -name 'config.json' -o -name 'model.safetensors.index.json' \
     -o -name '*.safetensors' \) -print0 \
  | sort -z | xargs -0 -r sha256sum > "$OUT_DIR/checkpoint.sha256"
if [[ ! -s "$OUT_DIR/checkpoint.sha256" ]]; then
  echo "No checkpoint files found under $MODEL_PATH" >&2
  exit 2
fi
"$SERVE_ENV/bin/python" -m pip freeze --all > "$OUT_DIR/environment.freeze.txt"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 2
    done
  fi
}
trap cleanup EXIT

if [[ "$BACKEND" == vllm ]]; then
  nohup "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$MEM_FRACTION" --moe-backend "$VLLM_MOE_BACKEND" \
    --enforce-eager \
    > "$SERVER_LOG" 2>&1 < /dev/null &
else
  nohup "$SERVE_ENV/bin/sglang" serve \
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" --tp-size "$TP_SIZE" \
    --context-length "$MAX_MODEL_LEN" --mem-fraction-static "$MEM_FRACTION" \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled --enable-metrics \
    > "$SERVER_LOG" 2>&1 < /dev/null &
fi
server_pid=$!
printf '%s\n' "$server_pid" > "$PID_FILE"

status="startup_timeout"
for _ in $(seq 1 240); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    status="healthy"
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    status="process_exited"
    break
  fi
  sleep 5
done

grep -Ein \
  'parameter.*not found|missing parameter|missing key|unexpected key|not loaded|skipped weight' \
  "$SERVER_LOG" > "$OUT_DIR/weight_coverage_warnings.txt" || true

if [[ "$status" == healthy ]]; then
  if ROOT_URL="http://127.0.0.1:$PORT" SERVED_NAME="$SERVED_NAME" \
    OUT_DIR="$OUT_DIR/acceptance" "$ROOT/serving/acceptance.sh"; then
    if [[ -s "$OUT_DIR/weight_coverage_warnings.txt" ]]; then
      status="rejected_weight_coverage_warning"
    else
      status="accepted"
    fi
    if [[ "$status" == accepted ]] && (( SMOKE_REQUESTS > 0 )); then
      if "$SERVE_ENV/bin/python" "$ROOT/clients/smoke.py" \
        --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_NAME" \
        --input-tokens "$SMOKE_INPUT_TOKENS" \
        --output-tokens "$SMOKE_OUTPUT_TOKENS" \
        --concurrency "$SMOKE_CONCURRENCY" --requests "$SMOKE_REQUESTS" \
        --seed "$SMOKE_SEED" --output "$OUT_DIR/smoke.jsonl" \
        --summary "$OUT_DIR/smoke.summary.json"; then
        status="accepted"
      else
        status="smoke_failed"
      fi
    fi
    if [[ "$status" == accepted && -n "$QUALITY_MANIFEST" ]]; then
      if [[ ! -f "$QUALITY_MANIFEST" ]]; then
        echo "QUALITY_MANIFEST does not exist: $QUALITY_MANIFEST" >&2
        status="quality_manifest_missing"
      elif ! "$SERVE_ENV/bin/python" "$ROOT/clients/quality_eval.py" \
        --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_NAME" \
        --manifest "$QUALITY_MANIFEST" --concurrency "$QUALITY_CONCURRENCY" \
        --timeout "$QUALITY_TIMEOUT" \
        --seed "$SMOKE_SEED" --output "$OUT_DIR/quality.results.jsonl" \
        --summary "$OUT_DIR/quality.summary.json"; then
        status="quality_eval_failed"
      fi
    fi
  else
    status="acceptance_failed"
  fi
fi

python3 - "$STATUS_FILE" "$BACKEND" "$MODEL_PATH" "$status" "$server_pid" \
  "$SMOKE_REQUESTS" "$SMOKE_CONCURRENCY" "$SMOKE_SEED" \
  "$QUALITY_MANIFEST" "$QUALITY_CONCURRENCY" "$VLLM_MOE_BACKEND" <<'PY'
import json, pathlib, sys
path, backend, model, status, pid, requests, concurrency, seed, quality, quality_c, moe_backend = sys.argv[1:]
payload = {"backend": backend, "model_path": model, "status": status,
           "server_pid": int(pid), "adapter_or_override_used": False,
           "vllm_moe_backend": moe_backend if backend == "vllm" else None,
           "smoke": {"requests": int(requests), "concurrency": int(concurrency),
                     "seed": int(seed)},
           "quality": {"manifest": quality or None,
                       "concurrency": int(quality_c) if quality else None}}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
PY

nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader > "$OUT_DIR/gpu_after.csv"
[[ "$status" == accepted ]]
