#!/usr/bin/env bash
# 作用：按冻结顺序运行 Phase 8 独立 selector Gate。
set -euo pipefail

# 在冻结 selector 后顺序执行 Phase 8 独立 Gate。脚本只终止自己启动的 incumbent
# 进程组；正式四候选 runner 也只管理各自创建的服务进程组。

if [[ $# -ne 3 ]]; then
  echo "用法: $0 <冻结代码目录> <冻结 selector.json> <输出根目录>" >&2
  exit 2
fi

CODE_DIR="$(cd "$1" && pwd)"
SELECTOR_CONFIG="$(realpath "$2")"
OUTPUT_ROOT="$3"
PYTHON_BIN="/data/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129/bin/python3"
VLLM_BIN="/data/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129/bin/vllm"
PLAN="$CODE_DIR/configs/experiments/phase8_selector_independent_v1.json"
MATRIX="$CODE_DIR/configs/workloads/phase8_selector_independent_v1.json"
TRAINING_AGGREGATE="/data/models/test/qtopomoe_phase8_selector_training_v2/aggregate_with_state.json"
MODEL_PATH="/data/models/test/models/Qwen--Qwen3.6-35B-A3B-FP8/snapshots/master"
SERVED_MODEL="qtopomoe-phase8-selector-incumbent-fp8"
PORT=31540
INCUMBENT_PID=""

for path in "$PYTHON_BIN" "$VLLM_BIN" "$PLAN" "$MATRIX" \
  "$SELECTOR_CONFIG" "$TRAINING_AGGREGATE"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少冻结输入: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/manager.log") 2>&1

stop_incumbent() {
  if [[ -z "$INCUMBENT_PID" ]] || ! kill -0 "$INCUMBENT_PID" 2>/dev/null; then
    return
  fi
  kill -TERM -- "-$INCUMBENT_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! kill -0 "$INCUMBENT_PID" 2>/dev/null; then
      wait "$INCUMBENT_PID" 2>/dev/null || true
      INCUMBENT_PID=""
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$INCUMBENT_PID" 2>/dev/null || true
  wait "$INCUMBENT_PID" 2>/dev/null || true
  INCUMBENT_PID=""
}
trap stop_incumbent EXIT INT TERM

if ss -ltn | grep -q ":$PORT "; then
  echo "端口 $PORT 已被占用，拒绝启动" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "存在 GPU 计算进程，独立 Gate 要求八卡空闲" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/predecision"
INCUMBENT_LOG="$OUTPUT_ROOT/predecision/incumbent.server.log"
setsid env \
  CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PATH="$(dirname "$VLLM_BIN"):$PATH" \
  numactl --cpunodebind=0 --membind=0 \
  "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 2 --data-parallel-size 1 \
  --max-model-len 65536 --max-num-seqs 128 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --moe-backend triton >"$INCUMBENT_LOG" 2>&1 &
INCUMBENT_PID=$!
echo "$INCUMBENT_PID" > "$OUTPUT_ROOT/predecision/incumbent.pgid"

ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$INCUMBENT_PID" 2>/dev/null; then
    echo "incumbent 在健康检查前退出" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "incumbent 健康检查超时" >&2
  exit 1
fi
if ! grep -q "Using TRITON Fp8 MoE backend" "$INCUMBENT_LOG"; then
  echo "incumbent backend 日志 Gate 失败" >&2
  exit 1
fi
CACHE_BLOCK_TOKENS="$(grep -oE 'Setting attention block size to [0-9]+ tokens' \
  "$INCUMBENT_LOG" | tail -1 | grep -oE '[0-9]+' || true)"
if [[ -z "$CACHE_BLOCK_TOKENS" ]]; then
  echo "无法从日志获取 prefix-cache block size" >&2
  exit 1
fi

"$PYTHON_BIN" "$CODE_DIR/scripts/run_phase8_predecision_windows.py" \
  --matrix "$MATRIX" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$SERVED_MODEL" \
  --tokenizer "$MODEL_PATH" \
  --incumbent-candidate-id fp8_tp2_pix01_triton \
  --cache-block-tokens "$CACHE_BLOCK_TOKENS" \
  --output-root "$OUTPUT_ROOT/predecision" \
  --python-bin "$PYTHON_BIN" \
  --selector-window-requests 16 --timeout 900

stop_incumbent
trap - EXIT INT TERM

for _ in $(seq 1 120); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    break
  fi
  sleep 1
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "incumbent 进程组已终止，但 GPU 资源未在等待窗口内释放" >&2
  exit 1
fi

"$PYTHON_BIN" "$CODE_DIR/scripts/run_phase8_repeated.py" \
  --plan "$PLAN" --output-root "$OUTPUT_ROOT/repeated"

"$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_phase8_repeated.py" \
  --run-root "$OUTPUT_ROOT/repeated" \
  --output "$OUTPUT_ROOT/aggregate.json" --bootstrap-samples 10000

"$PYTHON_BIN" "$CODE_DIR/scripts/attach_phase8_predecision_state.py" \
  --aggregate "$OUTPUT_ROOT/aggregate.json" \
  --predecision-states "$OUTPUT_ROOT/predecision/predecision_states.json" \
  --output "$OUTPUT_ROOT/aggregate_with_state.json"

"$PYTHON_BIN" "$CODE_DIR/scripts/evaluate_phase8_independent_selector.py" \
  --training-aggregate "$TRAINING_AGGREGATE" \
  --test-aggregate "$OUTPUT_ROOT/aggregate_with_state.json" \
  --selector-config "$SELECTOR_CONFIG" \
  --output "$OUTPUT_ROOT/independent_gate.json"

echo "Phase 8 selector 独立 Gate 已完成：$OUTPUT_ROOT/independent_gate.json"
