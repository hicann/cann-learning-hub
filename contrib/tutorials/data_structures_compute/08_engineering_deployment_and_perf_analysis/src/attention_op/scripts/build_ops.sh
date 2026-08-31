#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPS_DIR="$LAB_DIR/custom_ops/generated/AttentionCustom"

echo "=== Building AttentionCustom operator ==="
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
echo "=== Installing operator package to \${HOME}/vendors/customize ==="
chmod +x "$RUN_FILE"
bash "$RUN_FILE" --install-path="${HOME}" --quiet

echo "=== Operator package installed ==="
