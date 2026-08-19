#!/usr/bin/env bash
set -euo pipefail
command -v cmake; command -v python3; command -v npu-smi || true
[[ -n "${ASCEND_HOME_PATH:-}" || -d /usr/local/Ascend/ascend-toolkit/latest || -d /home/user/Ascend/ascend-toolkit/cann-8.5.0 ]] || { echo "[ERROR] CANN toolkit not found"; exit 1; }
python3 -c 'import torch; print("torch", torch.__version__)'
