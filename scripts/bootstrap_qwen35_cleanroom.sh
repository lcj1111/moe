#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/qwen35_cleanroom.env"

BASE="$QTOPOMOE_CLEANROOM_ROOT"
DOWNLOADS="$BASE/downloads"
SOURCES="$BASE/src"
VENVS="$BASE/venvs"
MANIFESTS="$BASE/manifests"
mkdir -p "$DOWNLOADS" "$SOURCES" "$VENVS" "$MANIFESTS"

download_and_verify() {
  local url="$1" output="$2" expected="$3"
  if [[ ! -f "$output" ]]; then
    curl -fL --retry 5 --retry-all-errors --connect-timeout 15 \
      --max-time 1800 -o "$output" "$url"
  fi
  printf '%s  %s\n' "$expected" "$output" | sha256sum --check --status
}

VLLM_TARBALL="$DOWNLOADS/vllm-$QTOPOMOE_VLLM_COMMIT.tar.gz"
VLLM_WHEEL="$DOWNLOADS/$QTOPOMOE_VLLM_WHEEL"
download_and_verify \
  "https://codeload.github.com/vllm-project/vllm/tar.gz/$QTOPOMOE_VLLM_COMMIT" \
  "$VLLM_TARBALL" "$QTOPOMOE_VLLM_SOURCE_SHA256"
download_and_verify \
  "https://wheels.vllm.ai/$QTOPOMOE_VLLM_COMMIT/${QTOPOMOE_VLLM_WHEEL/+/%2B}" \
  "$VLLM_WHEEL" "$QTOPOMOE_VLLM_WHEEL_SHA256"

VLLM_SOURCE="$SOURCES/vllm-$QTOPOMOE_VLLM_COMMIT"
if [[ ! -d "$VLLM_SOURCE" ]]; then
  tar -xzf "$VLLM_TARBALL" -C "$SOURCES"
fi
test -f "$VLLM_SOURCE/pyproject.toml"

VLLM_ENV="$VENVS/vllm-$QTOPOMOE_VLLM_COMMIT"
if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  python3 -m venv "$VLLM_ENV"
fi
"$VLLM_ENV/bin/python" -m pip install --upgrade \
  "pip==$QTOPOMOE_BOOTSTRAP_PIP_VERSION"
"$VLLM_ENV/bin/python" -m pip install "$VLLM_WHEEL"
"$VLLM_ENV/bin/python" -m pip check
"$VLLM_ENV/bin/python" -m pip freeze --all \
  > "$MANIFESTS/vllm-$QTOPOMOE_VLLM_COMMIT.freeze.txt"

SGLANG_TARBALL="$DOWNLOADS/sglang-$QTOPOMOE_SGLANG_COMMIT.tar.gz"
download_and_verify \
  "https://codeload.github.com/sgl-project/sglang/tar.gz/$QTOPOMOE_SGLANG_COMMIT" \
  "$SGLANG_TARBALL" "$QTOPOMOE_SGLANG_SOURCE_SHA256"

SGLANG_SOURCE="$SOURCES/sglang-$QTOPOMOE_SGLANG_COMMIT"
if [[ ! -d "$SGLANG_SOURCE" ]]; then
  tar -xzf "$SGLANG_TARBALL" -C "$SOURCES"
fi
test -f "$SGLANG_SOURCE/python/pyproject.toml"

SGLANG_ENV="$VENVS/sglang-$QTOPOMOE_SGLANG_COMMIT"
if [[ ! -x "$SGLANG_ENV/bin/python" ]]; then
  python3 -m venv "$SGLANG_ENV"
fi
"$SGLANG_ENV/bin/python" -m pip install --upgrade \
  "pip==$QTOPOMOE_BOOTSTRAP_PIP_VERSION"
SGLANG_BUILD_RUST_EXTS="$QTOPOMOE_SGLANG_BUILD_RUST_EXTS" \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$QTOPOMOE_SGLANG_VERSION" \
  "$SGLANG_ENV/bin/python" -m pip install "$SGLANG_SOURCE/python"
"$SGLANG_ENV/bin/python" -m pip check
"$SGLANG_ENV/bin/python" -m pip freeze --all \
  > "$MANIFESTS/sglang-$QTOPOMOE_SGLANG_COMMIT.freeze.txt"

"$VLLM_ENV/bin/python" - <<'PY'
import torch, transformers, vllm
print({"vllm": vllm.__version__, "torch": torch.__version__,
       "cuda": torch.version.cuda, "transformers": transformers.__version__,
       "cuda_available": torch.cuda.is_available()})
PY
"$SGLANG_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import sglang, torch, transformers
print({"sglang": sglang.__version__, "sglang_kernel": md.version("sglang-kernel"),
       "flashinfer": md.version("flashinfer-python"), "torch": torch.__version__,
       "cuda": torch.version.cuda, "transformers": transformers.__version__,
       "cuda_available": torch.cuda.is_available()})
PY
