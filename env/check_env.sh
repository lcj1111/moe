#!/usr/bin/env bash
# 作用：检查项目路径、CUDA、Python 环境和必要资产是否满足运行条件。
set -uo pipefail

env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project.env
source "${env_dir}/project.env"
if [[ -f "${env_dir}/local.env" ]]; then
  # shellcheck source=/dev/null
  source "${env_dir}/local.env"
fi

failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

check_dir() {
  local path=$1 label=$2
  [[ -d "$path" ]] && pass "$label: $path" || fail "$label missing: $path"
}

check_file() {
  local path=$1 label=$2
  [[ -r "$path" ]] && pass "$label: $path" || fail "$label missing/unreadable: $path"
}

check_cmd() {
  local cmd=$1
  command -v "$cmd" >/dev/null 2>&1 && pass "command $cmd: $(command -v "$cmd")" || fail "command missing: $cmd"
}

check_dir "$QTOPOMOE_ROOT" "project root"
check_dir "$QTOPOMOE_ARTIFACTS" "artifact root"
check_dir "$CUDA_HOME" "CUDA home"
check_file "$CUDACXX" "nvcc"
check_file "$QTOPOMOE_SGLANG_VENV/bin/python" "SGLang Python"
check_file "$QTOPOMOE_VLLM_VENV/bin/python" "vLLM Python"
check_file "$QTOPOMOE_FP8_MODEL/config.json" "FP8 checkpoint"
check_file "$QTOPOMOE_NVFP4_MODEL/config.json" "NVFP4 checkpoint"
check_file "$QTOPOMOE_ARTIFACTS/raw/20260807T120000Z_nccl_formal_p2p/nccl/statistics.json" "Phase 0 P2P formal statistics"
check_dir "$QTOPOMOE_NCCL_TESTS" "nccl-tests build"

for cmd in nvidia-smi numactl git cmake; do
  check_cmd "$cmd"
done

if [[ "$NCCL_IB_DISABLE" == "1" ]]; then
  pass "NCCL_IB_DISABLE=1"
else
  fail "NCCL_IB_DISABLE must default to 1 on this host"
fi

if timeout 30s "$QTOPOMOE_SGLANG_VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert torch.cuda.get_device_capability(0) == (12, 0)
print("SGLang torch", torch.__version__, "CUDA", torch.version.cuda,
      "NCCL", torch.cuda.nccl.version(), "GPUs", torch.cuda.device_count())
PY
then
  pass "SGLang PyTorch CUDA/SM120"
else
  fail "SGLang PyTorch CUDA/SM120 validation"
fi

if timeout 30s "$QTOPOMOE_VLLM_VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
assert torch.cuda.get_device_capability(0) == (12, 0)
print("vLLM torch", torch.__version__, "CUDA", torch.version.cuda,
      "NCCL", torch.cuda.nccl.version(), "GPUs", torch.cuda.device_count())
PY
then
  pass "vLLM PyTorch CUDA/SM120"
else
  fail "vLLM PyTorch CUDA/SM120 validation"
fi

if timeout 20s nvidia-smi \
  --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader; then
  pass "nvidia-smi query"
else
  warn "nvidia-smi query timed out"
fi

printf '\nSummary: failures=%d warnings=%d\n' "$failures" "$warnings"
(( failures == 0 ))
