#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SERVED_NAME:?SERVED_NAME is required}"
: "${GPU_IDS:?GPU_IDS is required}"
: "${TP_SIZE:?TP_SIZE is required}"
: "${PORT:?PORT is required}"
: "${SERVE_ENV:?SERVE_ENV is required}"

BACKEND="${BACKEND:-sglang}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"
export PATH="$SERVE_ENV/bin:$PATH"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -x "$SERVE_ENV/bin/python" ]]; then
  echo "SERVE_ENV has no python: $SERVE_ENV" >&2
  exit 2
fi

case "$BACKEND" in
  sglang)
    args=(serve
      --model-path "$MODEL_PATH"
      --served-model-name "$SERVED_NAME"
      --host "${HOST:-0.0.0.0}"
      --port "$PORT"
      --tp-size "$TP_SIZE"
      --enable-metrics
      --dtype "${DTYPE:-auto}"
      --mem-fraction-static "${MEM_FRACTION_STATIC:-0.82}"
    )
    if [[ -n "${QUANTIZATION:-}" ]]; then
      args+=(--quantization "$QUANTIZATION")
    fi
    if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
      args+=(--disable-radix-cache)
    fi
    exec "$SERVE_ENV/bin/sglang" "${args[@]}"
    ;;
  vllm)
    args=(serve "$MODEL_PATH"
      --served-model-name "$SERVED_NAME"
      --host "${HOST:-0.0.0.0}"
      --port "$PORT"
      --tensor-parallel-size "$TP_SIZE"
      --dtype "${DTYPE:-auto}"
    )
    if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
      args+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
      read -r -a extra_args <<< "$EXTRA_ARGS"
      args+=("${extra_args[@]}")
    fi
    exec "$SERVE_ENV/bin/vllm" "${args[@]}"
    ;;
  *)
    echo "Unsupported BACKEND=$BACKEND (expected sglang or vllm)" >&2
    exit 2
    ;;
esac
