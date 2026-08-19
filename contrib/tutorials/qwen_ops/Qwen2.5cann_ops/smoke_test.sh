#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/setup_cannlab_env.sh"

projects=(
  RmsNormBaselineExperiment RmsNormOptimizedExperiment
  RopeBaselineExperiment RopeOptimizedExperiment
  SwiGluBaselineExperiment SwiGluOptimizedExperiment
  GemmBaselineExperiment GemmOptimizedExperiment
  GqaAttentionBaselineExperiment GqaAttentionOptimizedExperiment
)

for project in "${projects[@]}"; do
    echo "===== Testing $project ====="
    (
        cd "$QWEN_OPS_CODE_ROOT/$project"
        "$PYTHON_BIN" tests/test_torch_op.py
    )
done

echo "===== Baseline five-operator integration ====="
bash "$QWEN_OPS_CODE_ROOT/Qwen2.5BaselineIntegrationExperiment/run.sh" \
  --model "$QWEN_OPS_MODEL_PATH" --repeat 1

echo "===== Optimized five-operator integration ====="
bash "$QWEN_OPS_CODE_ROOT/Qwen2.5OptimizedIntegrationExperiment/run.sh" \
  --model "$QWEN_OPS_MODEL_PATH" --repeat 1

echo "[PASS] qwen_ops smoke test completed"
