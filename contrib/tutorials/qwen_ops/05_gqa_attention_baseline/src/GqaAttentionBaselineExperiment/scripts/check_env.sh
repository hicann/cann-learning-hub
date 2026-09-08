#!/usr/bin/env bash
set -euo pipefail

QWEN_OPS_COURSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$QWEN_OPS_COURSE_ROOT/scripts/setup_cannlab_env.sh"
command -v cmake; command -v python3; command -v npu-smi || true
[[ -n "${ASCEND_HOME_PATH:-}" || -d /usr/local/Ascend/ascend-toolkit/latest ]] || { echo "[ERROR] CANN toolkit not found; set ASCEND_HOME_PATH"; exit 1; }
python3 -c 'import torch; print("torch", torch.__version__)'
