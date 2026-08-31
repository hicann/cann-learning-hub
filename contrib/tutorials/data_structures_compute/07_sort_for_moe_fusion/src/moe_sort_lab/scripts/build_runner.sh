#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$SCRIPT_DIR/env_custom_opp.sh" ]; then source "$SCRIPT_DIR/env_custom_opp.sh"; fi

cd "$PROJECT_DIR/aclnn_runner"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "Built main_benchmark and main_full_pipeline_benchmark under $PROJECT_DIR/aclnn_runner/build"
