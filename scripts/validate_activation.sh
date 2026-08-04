#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# shellcheck source=../env/activate.sh
source env/activate.sh

qtopomoe_use_sglang
python - <<'PY'
import torch
print("ACTIVE_SGLANG", torch.__version__, torch.version.cuda)
PY
deactivate

qtopomoe_new_run bootstrap_validation
test -d "$QTOPOMOE_RUN_DIR/server"
test -d "$QTOPOMOE_RUN_DIR/client"
echo "RUN_DIRECTORY_OK $QTOPOMOE_RUN_DIR"

qtopomoe_use_vllm
python - <<'PY'
import torch
print("ACTIVE_VLLM", torch.__version__, torch.version.cuda)
PY
deactivate

echo "ACTIVATION_VALIDATION_PASS"
