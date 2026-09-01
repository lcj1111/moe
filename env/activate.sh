#!/usr/bin/env bash
# 作用：加载项目环境变量和可选的本机覆盖配置。

_qtopomoe_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project.env
source "${_qtopomoe_env_dir}/project.env"
if [[ -f "${_qtopomoe_env_dir}/local.env" ]]; then
  # shellcheck source=/dev/null
  source "${_qtopomoe_env_dir}/local.env"
fi
unset _qtopomoe_env_dir

qtopomoe_use_sglang() {
  if [[ ! -f "${QTOPOMOE_SGLANG_VENV}/bin/activate" ]]; then
    echo "Missing SGLang environment: ${QTOPOMOE_SGLANG_VENV}" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "${QTOPOMOE_SGLANG_VENV}/bin/activate"
  export PATH="${CUDA_HOME}/bin:${VIRTUAL_ENV}/bin:${PATH}"
  echo "Using SGLang environment: ${VIRTUAL_ENV}"
}

qtopomoe_use_vllm() {
  if [[ ! -f "${QTOPOMOE_VLLM_VENV}/bin/activate" ]]; then
    echo "Missing vLLM environment: ${QTOPOMOE_VLLM_VENV}" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "${QTOPOMOE_VLLM_VENV}/bin/activate"
  export PATH="${CUDA_HOME}/bin:${VIRTUAL_ENV}/bin:${PATH}"
  echo "Using vLLM environment: ${VIRTUAL_ENV}"
}

qtopomoe_new_run() {
  local config_id="${1:-manual}"
  if [[ ! "$config_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "config_id may contain only letters, numbers, dot, underscore, and hyphen" >&2
    return 2
  fi
  export QTOPOMOE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${config_id}"
  export QTOPOMOE_RUN_DIR="${QTOPOMOE_ARTIFACTS}/raw/${QTOPOMOE_RUN_ID}"
  install -d "${QTOPOMOE_RUN_DIR}"/{server,client,metrics,env,profiles}
  printf 'run_id=%s\nrun_dir=%s\n' "$QTOPOMOE_RUN_ID" "$QTOPOMOE_RUN_DIR"
}

qtopomoe_gpu_status() {
  nvidia-smi \
    --query-gpu=index,uuid,pstate,memory.used,memory.free,utilization.gpu \
    --format=csv
  echo "-- compute processes --"
  nvidia-smi \
    --query-compute-apps=pid,process_name,gpu_uuid,used_memory \
    --format=csv,noheader 2>/dev/null || true
}

echo "Q-TopoMoE environment loaded"
echo "  root=${QTOPOMOE_ROOT}"
echo "  artifacts=${QTOPOMOE_ARTIFACTS}"
echo "  cuda=${CUDA_HOME}"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "Choose a framework with qtopomoe_use_sglang or qtopomoe_use_vllm."
