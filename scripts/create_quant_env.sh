#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/project.env"

python3 -m venv "$QTOPOMOE_QUANT_ENV"
SITE_DIR="$($QTOPOMOE_QUANT_ENV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' \
  'import sys' \
  'sys.path.append("/data/models/test/.sglang_env/lib/python3.12/site-packages")' \
  > "$SITE_DIR/sitecustomize.py"

"$QTOPOMOE_QUANT_ENV/bin/python" -m pip install --upgrade pip
# Reuse the already validated CUDA 13/Python 3.12 torch and Transformers from
# the serving environment without modifying that environment. Quant tooling
# itself is installed into the independent quant venv.
"$QTOPOMOE_QUANT_ENV/bin/pip" install --no-deps \
  accelerate datasets safetensors llmcompressor compressed-tensors

export PYTHONPATH="/data/models/test/.sglang_env/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
"$QTOPOMOE_QUANT_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
import transformers
import compressed_tensors
import llmcompressor

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
for name in ("accelerate", "datasets", "safetensors", "llmcompressor", "compressed-tensors"):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, "MISSING")
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

"$QTOPOMOE_QUANT_ENV/bin/pip" check
mkdir -p "$ROOT/env/requirements-lock"
"$QTOPOMOE_QUANT_ENV/bin/pip" freeze --all > "$ROOT/env/requirements-lock/quant.txt"
