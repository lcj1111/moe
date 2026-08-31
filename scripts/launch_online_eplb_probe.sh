#!/usr/bin/env bash
# 作用：启动带 placement 补丁的在线 EPLB 探测服务。
set -euo pipefail

RUN_DIR=${RUN_DIR:-/data/models/test/qtopomoe_online_eplb_probe_v1}
PORT=${PORT:-31680}
PLAN=${PLAN:-/data/models/test/qtopomoe_traces/v1/nvfp4_redhat_full/runtime_eplb/physical_to_logical_map.json}
VENV=${VENV:-/data/moe/.runtime/cleanroom/venvs/vllm-33c50587d2679ba9bacc2a51ae19901f7eb3a129}
PATCH_DIR=${PATCH_DIR:-/data/moe/runtime_patches/qtopomoe_eplb}
MODEL=${MODEL:-/data/models/test/redhatai_qwen36_nvfp4}

mkdir -p "$RUN_DIR"
if [[ -s "$RUN_DIR/server.pid" ]] && kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
  echo "probe is already running: PID $(cat "$RUN_DIR/server.pid")" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export QTOPOMOE_EPLB_PLAN="$PLAN"
export QTOPOMOE_EPLB_DEFER_CALLS=${QTOPOMOE_EPLB_DEFER_CALLS:-0}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nohup "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name qtopomoe-nvfp4-eplb-probe \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --enable-prompt-tokens-details \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"window_size":16,"step_interval":16,"num_redundant_experts":0,"log_balancedness":true,"log_balancedness_interval":8,"use_async":false,"communicator":"torch_nccl"}' \
  >"$RUN_DIR/server.log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$RUN_DIR/server.pid"
"$VENV/bin/python" - <<PY
import json, time
from pathlib import Path
Path("$RUN_DIR/launch.json").write_text(json.dumps({
    "pid": $pid,
    "port": $PORT,
    "plan": "$PLAN",
    "model": "$MODEL",
    "started_unix": time.time(),
    "purpose": "NVFP4 native synchronous EPLB compatibility probe",
}, indent=2) + "\n")
PY
echo "$pid"
