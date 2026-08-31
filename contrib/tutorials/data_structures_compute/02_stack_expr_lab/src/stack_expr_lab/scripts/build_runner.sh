#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_DIR="$LAB_DIR/aclnn_runner"

echo "=== Building Benchmark Runner ==="
echo "RUNNER_DIR: $RUNNER_DIR"

cd "$RUNNER_DIR"

# Clean previous build
rm -rf build
mkdir -p build
cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo ""
echo "=== Runner build completed ==="
ls -la main_stack_benchmark
