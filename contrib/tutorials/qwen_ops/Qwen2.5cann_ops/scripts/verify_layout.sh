#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
projects=(
  RopeBaselineExperiment RopeOptimizedExperiment
  GemmBaselineExperiment GemmOptimizedExperiment
  GqaAttentionBaselineExperiment GqaAttentionOptimizedExperiment
  RmsNormBaselineExperiment RmsNormOptimizedExperiment
  SwiGluBaselineExperiment SwiGluOptimizedExperiment
)

for project in "${projects[@]}"; do
  for path in CMakeLists.txt scripts tests op_kernel torch_extension; do
    [[ -e "$ROOT/$project/$path" ]] || { echo "[FAIL] missing $project/$path"; exit 1; }
  done
  echo "[PASS] $project source layout present"
done

python3 -m py_compile \
  "$ROOT/Qwen2.5BaselineIntegrationExperiment/qwen2_5_five_ops_benchmark.py" \
  "$ROOT/Qwen2.5OptimizedIntegrationExperiment/qwen2_5_five_ops_benchmark.py"
echo "[PASS] integration scripts parse successfully"
