#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env/project.env
source "${script_dir}/../env/project.env"

snapshot="${QTOPOMOE_ARTIFACTS}/manifests/env_$(date -u +%Y%m%dT%H%M%SZ)"
install -d "$snapshot"

hostname > "$snapshot/hostname.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$snapshot/created_at_utc.txt"
uname -a > "$snapshot/uname.txt"
cat /proc/cmdline > "$snapshot/kernel_cmdline.txt"
nvidia-smi -q > "$snapshot/nvidia-smi-q.txt"
"$CUDACXX" --version > "$snapshot/nvcc.txt"

"$QTOPOMOE_SGLANG_VENV/bin/python" -VV > "$snapshot/sglang-python.txt" 2>&1
"$QTOPOMOE_SGLANG_VENV/bin/python" -m pip freeze --all > "$snapshot/sglang-pip-freeze.txt"
"$QTOPOMOE_VLLM_VENV/bin/python" -VV > "$snapshot/vllm-python.txt" 2>&1
"$QTOPOMOE_VLLM_VENV/bin/python" -m pip freeze --all > "$snapshot/vllm-pip-freeze.txt"

for model in "$QTOPOMOE_FP8_MODEL" "$QTOPOMOE_NVFP4_MODEL"; do
  printf 'path=%s\n' "$(readlink -f "$model")"
  sha256sum "$model/config.json"
  if [[ -f "$model/model.safetensors.index.json" ]]; then
    sha256sum "$model/model.safetensors.index.json"
  fi
done > "$snapshot/model-identities.txt"

sha256sum "$QTOPOMOE_PHASE0/nccl/formal5/formal5-statistics.csv" \
  > "$snapshot/phase0-formal5.sha256"

echo "Snapshot written to $snapshot"
