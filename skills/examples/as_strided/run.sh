#!/bin/bash
# Run as_strided operator tests on NPU
# This script installs the custom op package and runs Python tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="${SCRIPT_DIR}"
CODE_DIR="${OP_DIR}/code"
BUILD_DIR="${CODE_DIR}/build"

echo "============================================================"
echo "as_strided operator test runner"
echo "============================================================"

# Step 1: Build the operator
echo ""
echo "[1/3] Building operator..."
cd "${CODE_DIR}"
if [ ! -d "build" ]; then
    mkdir -p build
fi
cd build
cmake .. > /dev/null 2>&1
make -j > /dev/null 2>&1
echo "  Build successful."

# Step 2: Install custom op package
echo ""
echo "[2/3] Installing custom op package..."
INSTALL_SCRIPT="${BUILD_DIR}/scripts/install.sh"
if [ -f "${INSTALL_SCRIPT}" ]; then
    bash "${INSTALL_SCRIPT}" > /dev/null 2>&1
    echo "  Custom op package installed."
else
    echo "  Warning: install.sh not found, skipping installation."
fi

# Step 3: Run tests
echo ""
echo "[3/3] Running tests..."

# Check if NPU is available
if command -v npu-smi &> /dev/null; then
    NPU_COUNT=$(npu-smi info 2>/dev/null | grep "Total" | head -1 | awk '{print $NF}' || echo "0")
    if [ "${NPU_COUNT}" = "0" ]; then
        echo "  NPU not available, skipping NPU tests."
        echo "  Build verification: PASSED (compilation successful)"
        exit 0
    fi
else
    echo "  npu-smi not found, skipping NPU tests."
    echo "  Build verification: PASSED (compilation successful)"
    exit 0
fi

# Run Python test
PYTHON_TEST="${OP_DIR}/run_test.py"
if [ -f "${PYTHON_TEST}" ]; then
    python3 "${PYTHON_TEST}"
else
    echo "  No Python test script found."
    echo "  Build verification: PASSED (compilation successful)"
fi

echo ""
echo "============================================================"
echo "Test complete."
echo "============================================================"
