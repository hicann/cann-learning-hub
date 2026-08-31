#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUNNER_DIR="$PROJECT_DIR/aclnn_runner"

if [ -f "$PROJECT_DIR/scripts/env_custom_opp.sh" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/scripts/env_custom_opp.sh"
fi

cd "$RUNNER_DIR"
rm -rf build
mkdir -p build
cd build
cmake ..
cmake --build . -j"$(nproc)"

echo "ACLNN runner built successfully:"
echo "  $RUNNER_DIR/build/main_tree_queue_benchmark"
