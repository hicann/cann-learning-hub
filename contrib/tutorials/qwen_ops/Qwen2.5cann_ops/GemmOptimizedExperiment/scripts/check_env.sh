#!/usr/bin/env bash
set -euo pipefail

echo "=== GEMM Optimized Environment Check ==="

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[OK] ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    echo "[OK] Found CANN at $ASCEND_HOME_PATH"
else
    echo "[FAIL] CANN not found. Set ASCEND_HOME_PATH."
    exit 1
fi

[[ -f /usr/local/Ascend/driver/bin/setenv.bash ]] && echo "[OK] NPU driver found" || echo "[WARN] driver setenv not found"
command -v cmake >/dev/null && echo "[OK] cmake available" || { echo "[FAIL] cmake required"; exit 1; }
python3 -c "import torch, torch_npu, numpy" >/dev/null && echo "[OK] Python torch/torch_npu/numpy available" || { echo "[FAIL] torch, torch_npu and numpy required"; exit 1; }
command -v npu-smi >/dev/null && npu-smi info | head -20 || echo "[WARN] npu-smi unavailable"

echo "=== Check finished ==="