#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_DIR="$LAB_DIR/aclnn_runner"

echo "=== Building aclnn runner ==="
cd "$RUNNER_DIR"
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
echo "=== Runner built: $RUNNER_DIR/build/main_attention_benchmark ==="
