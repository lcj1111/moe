#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
# shellcheck source=env/project.env
source env/project.env
if [[ -f env/local.env ]]; then
  # shellcheck source=/dev/null
  source env/local.env
fi

freeze_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
out="$QTOPOMOE_ARTIFACTS/manifests/freeze_${freeze_id}"
mkdir -p "$out" "$out/models" "$out/data" "$out/software" "$out/hardware"

capture() {
  local name=$1
  shift
  "$@" >"$out/$name" 2>&1 || printf 'command_failed=%s\n' "$*" >>"$out/$name"
}

date -u +%Y-%m-%dT%H:%M:%SZ >"$out/software/frozen_at_utc.txt"
hostname >"$out/software/hostname.txt"
uname -a >"$out/software/uname.txt"
cat /etc/os-release >"$out/software/os-release.txt"
cat /proc/cmdline >"$out/software/kernel-cmdline.txt"
nvidia-smi -q >"$out/hardware/nvidia-smi-q.txt"
nvidia-smi -L >"$out/hardware/nvidia-smi-L.txt"
nvidia-smi topo -m >"$out/hardware/topo-m.txt"
nvidia-smi topo -p2p r >"$out/hardware/topo-p2p-read.txt"
nvidia-smi topo -p2p w >"$out/hardware/topo-p2p-write.txt"
"$CUDACXX" --version >"$out/software/nvcc.txt"

"$QTOPOMOE_SGLANG_VENV/bin/python" -VV >"$out/software/sglang-python.txt" 2>&1
"$QTOPOMOE_SGLANG_VENV/bin/python" -m pip freeze --all >"$out/software/sglang-pip-freeze.txt"
"$QTOPOMOE_SGLANG_VENV/bin/python" -m pip check >"$out/software/sglang-pip-check.txt" 2>&1 || true
"$QTOPOMOE_SGLANG_VENV/bin/python" - <<'PY' >"$out/software/sglang-runtime.txt"
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version())
try:
    import sglang
    print("sglang", getattr(sglang, "__version__", "unknown"))
except Exception as e:
    print("sglang_import_error", repr(e))
PY
"$QTOPOMOE_SGLANG_VENV/bin/python" -m sglang.launch_server --help >"$out/software/sglang-launch-help.txt" 2>&1 || true

"$QTOPOMOE_VLLM_VENV/bin/python" -VV >"$out/software/vllm-python.txt" 2>&1
"$QTOPOMOE_VLLM_VENV/bin/python" -m pip freeze --all >"$out/software/vllm-pip-freeze.txt"
"$QTOPOMOE_VLLM_VENV/bin/python" -m pip check >"$out/software/vllm-pip-check.txt" 2>&1 || true
"$QTOPOMOE_VLLM_VENV/bin/vllm" serve --help >"$out/software/vllm-serve-help.txt" 2>&1 || \
  "$QTOPOMOE_VLLM_VENV/bin/python" -m vllm.entrypoints.cli.main serve --help >"$out/software/vllm-serve-help.txt" 2>&1 || true
"$QTOPOMOE_VLLM_VENV/bin/python" - <<'PY' >"$out/software/vllm-runtime.txt"
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version())
try:
    import vllm
    print("vllm", getattr(vllm, "__version__", "unknown"))
except Exception as e:
    print("vllm_import_error", repr(e))
PY

if [[ -d "$QTOPOMOE_ROOT/.git" ]]; then
  git -C "$QTOPOMOE_ROOT" status --short >"$out/software/project-git-status.txt"
  git -C "$QTOPOMOE_ROOT" log -1 --oneline >"$out/software/project-git-head.txt" 2>&1 || true
fi
if [[ -d "$QTOPOMOE_ROOT/third_party" ]]; then
  find "$QTOPOMOE_ROOT/third_party" -maxdepth 2 -type d -name .git -print \
    >"$out/software/third-party-repos.txt"
fi

freeze_model() {
  local id=$1 path=$2
  {
    printf 'model_id=%s\npath=%s\n' "$id" "$path"
    if [[ -f "$path/config.json" ]]; then
      printf 'config_status=present\n'
      stat -c 'config_stat=%n %s %y' "$path/config.json"
      sha256sum "$path/config.json"
    else
      printf 'config_status=missing\n'
    fi
    if [[ -f "$path/model.safetensors.index.json" ]]; then
      printf 'index_status=present\n'
      stat -c 'index_stat=%n %s %y' "$path/model.safetensors.index.json"
      sha256sum "$path/model.safetensors.index.json"
    else
      printf 'index_status=missing\n'
    fi
    if [[ -f "$path/config.json" ]]; then
      "$QTOPOMOE_SGLANG_VENV/bin/python" - "$path/config.json" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p))
keys=["architectures","model_type","num_hidden_layers","num_experts","quantization_config"]
print(json.dumps({k:d.get(k) for k in keys}, indent=2, default=str))
PY
    fi
  } >"$out/models/${id}.txt" 2>&1
}

freeze_model fp8 "$QTOPOMOE_FP8_MODEL"
freeze_model nvfp4 "$QTOPOMOE_NVFP4_MODEL"
if [[ -n "${QTOPOMOE_BF16_MODEL:-}" ]]; then
  freeze_model bf16 "$QTOPOMOE_BF16_MODEL"
else
  printf 'model_id=bf16\nstatus=path_not_configured\n' >"$out/models/bf16.txt"
fi
freeze_model w4a16 "$QTOPOMOE_W4A16_MODEL"

seed_source="/data/models/test/sglang_bench/sharegpt_seed.json"
seed_project="$QTOPOMOE_ROOT/data/manifests/sharegpt_seed.json"
if [[ -f "$seed_source" ]]; then
  install -d "$(dirname "$seed_project")"
  cp -p "$seed_source" "$seed_project"
  sha256sum "$seed_project" >"$out/data/sharegpt_seed.sha256"
  "$QTOPOMOE_SGLANG_VENV/bin/python" - "$seed_project" <<'PY' >"$out/data/sharegpt_seed_inventory.json"
import json, sys
p=sys.argv[1]
d=json.load(open(p))
print(json.dumps({
    "path": p,
    "type": type(d).__name__,
    "records": len(d) if isinstance(d, list) else None,
    "first_keys": sorted(d[0].keys()) if isinstance(d, list) and d else [],
}, indent=2))
PY
else
  printf '{"status":"missing","path":"%s"}\n' "$seed_source" >"$out/data/sharegpt_seed_inventory.json"
fi

if [[ -f "$QTOPOMOE_PHASE0/nccl/formal5/formal5-statistics.csv" ]]; then
  sha256sum "$QTOPOMOE_PHASE0/nccl/formal5/formal5-statistics.csv" \
    >"$out/hardware/phase0-reference.sha256"
fi
if [[ -f "$QTOPOMOE_ARTIFACTS/manifests/phase0_current.json" ]]; then
  sha256sum "$QTOPOMOE_ARTIFACTS/manifests/phase0_current.json" \
    >"$out/hardware/phase0-current.sha256"
fi

"$QTOPOMOE_SGLANG_VENV/bin/python" - "$out" <<'PY'
import json, os, sys
from pathlib import Path
out=Path(sys.argv[1])
def text(path):
    p=out / path
    return p.read_text(errors="replace") if p.exists() else ""
def status(path, marker):
    return marker in text(path)
manifest={
    "schema_version": 1,
    "freeze_id": out.name.removeprefix("freeze_"),
    "status": "software_models_and_smoke_data_frozen",
    "root": os.environ.get("QTOPOMOE_ROOT"),
    "artifact_root": os.environ.get("QTOPOMOE_ARTIFACTS"),
    "software": {
        "sglang_env": os.environ.get("QTOPOMOE_SGLANG_VENV"),
        "vllm_env": os.environ.get("QTOPOMOE_VLLM_VENV"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
        "project_git_head": text("software/project-git-head.txt").strip(),
        "third_party_repos_present": bool(text("software/third-party-repos.txt").strip()),
    },
    "models": {
        "fp8": {"status": "config_and_index_hash_recorded"},
        "nvfp4": {"status": "config_and_index_hash_recorded"},
        "bf16": {"status": "path_not_configured"},
        "w4a16": {"status": "not_generated"},
    },
    "data": {
        "sharegpt_seed": {
            "status": "copied_and_hashed",
            "inventory": text("data/sharegpt_seed_inventory.json").strip(),
        },
        "calibration_256": {"status": "not_frozen"},
        "evaluation_manifests": {"status": "not_frozen"},
    },
    "next_gate": "freeze_calibration_and_evaluation_data_before_quantization_formal_runs",
}
(out / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

printf '%s\n' "$out"
