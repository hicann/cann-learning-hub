#!/usr/bin/env bash
# QwenRoPeCustom 环境检查脚本
set -euo pipefail

echo "=== QwenRoPeCustom Environment Check ==="

# 1. CANN toolkit
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[OK] ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
elif [[ -d "/usr/local/Ascend/ascend-toolkit/latest" ]]; then
    echo "[OK] Found CANN at /usr/local/Ascend/ascend-toolkit/latest"
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
else
    echo "[FAIL] CANN not found. Set ASCEND_HOME_PATH or install CANN 8.x."
    exit 1
fi

# 2. NPU driver
if [[ -f /usr/local/Ascend/driver/bin/setenv.bash ]]; then
    echo "[OK] NPU driver found"
else
    echo "[FAIL] NPU driver not found"
    exit 1
fi

# 3. Python + NumPy
if python3 -c "import numpy" 2>/dev/null; then
    echo "[OK] Python3 + NumPy available"
else
    echo "[FAIL] Python3 + NumPy required"
    exit 1
fi

# 4. GCC 11
if g++-11 --version &>/dev/null; then
    echo "[OK] g++-11 available"
else
    echo "[WARN] g++-11 not found, may need CC/CXX override"
fi

# 5. CMake
if cmake --version &>/dev/null; then
    echo "[OK] cmake available"
else
    echo "[FAIL] cmake required"
    exit 1
fi

echo "=== All checks passed ==="
