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
    echo "===== Building $project ====="
    (
        cd "$QWEN_OPS_CODE_ROOT/$project"
        bash scripts/build.sh
    )
done

echo "[PASS] all ten operator projects built for $SOC_VERSION"
