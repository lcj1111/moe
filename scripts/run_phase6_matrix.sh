#!/usr/bin/env bash
# Phase 6 TP/DP/EP matrix runner: start vLLM, smoke 32 requests, benchmark.
#
# Usage:
#   BACKEND=vllm MODEL_PATH=<ckpt> GPU_IDS=0,1,2,3,4,5,6,7 \
#   TP_SIZE=8 DP_SIZE=1 EP_SIZE=0 PORT=31400 RUN_TAG=tp8_dp1 \
#   bash scripts/run_phase6_matrix.sh
#
# Each cell: start -> health -> smoke (32 requests, concurrency 8) ->
# summary -> shutdown.  Result JSON written under OUT_ROOT
# (/data/models/test/qtopomoe_phase6/<RUN_TAG>).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/qwen35_cleanroom.env"

BACKEND="${BACKEND:?BACKEND required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
GPU_IDS="${GPU_IDS:?GPU_IDS required}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-0}"
EPLB="${EPLB:-0}"
EPLB_CONFIG="${EPLB_CONFIG:-}"
PORT="${PORT:?PORT required}"
RUN_TAG="${RUN_TAG:?RUN_TAG required}"
OUT_ROOT="${OUT_ROOT:-/data/models/test/qtopomoe_phase6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
SMOKE_REQUESTS="${SMOKE_REQUESTS:-32}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-8}"
SMOKE_INPUT_TOKENS="${SMOKE_INPUT_TOKENS:-256}"
SMOKE_OUTPUT_TOKENS="${SMOKE_OUTPUT_TOKENS:-64}"
SMOKE_SEED="${SMOKE_SEED:-42}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-auto}"

SERVE_ENV="$QTOPOMOE_CLEANROOM_ROOT/venvs/vllm-$QTOPOMOE_VLLM_COMMIT"
RUN_DIR="$OUT_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR"
SERVER_LOG="$RUN_DIR/server.log"
PID_FILE="$RUN_DIR/server.pid"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PATH="$SERVE_ENV/bin:$PATH"
unset PYTHONPATH

test -d "$MODEL_PATH"
test -x "$SERVE_ENV/bin/vllm"
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "Port $PORT occupied" >&2
  exit 2
fi

EXTRA_ARGS=()
if (( DP_SIZE > 1 )); then
  EXTRA_ARGS+=(--data-parallel-size "$DP_SIZE")
fi
if (( EP_SIZE > 0 )); then
  EXTRA_ARGS+=(--enable-expert-parallel)
  EXTRA_ARGS+=(--all2all-backend allgather_reducescatter)
  EXTRA_ARGS+=(--expert-placement-strategy "${EP_PLACEMENT:-linear}")
fi
if (( EPLB > 0 )); then
  EXTRA_ARGS+=(--enable-eplb)
  if [[ -n "$EPLB_CONFIG" ]]; then
    EXTRA_ARGS+=(--eplb-config "$EPLB_CONFIG")
  fi
fi

nohup "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "qtopomoe-$RUN_TAG" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$MEM_FRACTION" --moe-backend "$VLLM_MOE_BACKEND" \
  "${EXTRA_ARGS[@]}" \
  > "$SERVER_LOG" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$PID_FILE"

cleanup() {
  kill -TERM "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 2
  done
}
trap cleanup EXIT

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

summary_file="$RUN_DIR/smoke.summary.json"
if [[ "$status" == healthy ]]; then
  if "$SERVE_ENV/bin/python" "$ROOT/clients/smoke.py" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "qtopomoe-$RUN_TAG" \
    --input-tokens "$SMOKE_INPUT_TOKENS" --output-tokens "$SMOKE_OUTPUT_TOKENS" \
    --concurrency "$SMOKE_CONCURRENCY" --requests "$SMOKE_REQUESTS" \
    --seed "$SMOKE_SEED" --output "$RUN_DIR/smoke.jsonl" \
    --summary "$summary_file"; then
    status="smoke_passed"
  else
    status="smoke_failed"
  fi
fi

{
  echo "{\"run_tag\": \"$RUN_TAG\", \"status\": \"$status\","
  echo " \"model_path\": \"$MODEL_PATH\", \"gpu_ids\": \"$GPU_IDS\","
  echo " \"tp\": $TP_SIZE, \"dp\": $DP_SIZE, \"ep\": $EP_SIZE,"
  echo " \"eplb\": $EPLB, \"eplb_config\": \"$EPLB_CONFIG\", \"port\": $PORT,"
  echo " \"smoke\": {\"requests\": $SMOKE_REQUESTS, \"concurrency\": $SMOKE_CONCURRENCY,"
  echo "  \"input_tokens\": $SMOKE_INPUT_TOKENS, \"output_tokens\": $SMOKE_OUTPUT_TOKENS}}"
} > "$RUN_DIR/meta.json"

echo "RUN_TAG=$RUN_TAG status=$status"
if [[ -f "$summary_file" ]]; then
  cat "$summary_file"
fi
[[ "$status" == smoke_passed ]]
