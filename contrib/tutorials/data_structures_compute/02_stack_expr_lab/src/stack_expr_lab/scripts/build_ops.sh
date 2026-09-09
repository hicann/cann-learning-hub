#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPS_DIR="$LAB_DIR/custom_ops/generated/StackExprOps"

echo "=== Building StackExprOps custom operators ==="
echo "OPS_DIR: $OPS_DIR"

cd "$OPS_DIR"

# Clean previous build
rm -rf build_out

# Run the build
bash build.sh

echo ""
echo "=== Build completed ==="
ls -la build_out/

# Find and install the .run package
RUN_FILE=$(find build_out -name "*.run" -type f | head -1)
if [ -z "$RUN_FILE" ]; then
    echo "ERROR: No .run file found in build_out"
    exit 1
fi

echo ""
echo "=== Installing operator package: $RUN_FILE ==="
chmod +x "$RUN_FILE"

# Install to custom OPP path
CUSTOM_OPP_DIR="${ASCEND_HOME_PATH}/opp/vendors/customize"
sudo mkdir -p "$CUSTOM_OPP_DIR"
sudo bash "$RUN_FILE" --install-path="${ASCEND_HOME_PATH}/opp/vendors" --install-for-all

echo "=== Operator package installed ==="
