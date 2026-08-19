#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
execute=0
[[ "${1:-}" == "--execute" ]] && execute=1

projects=(
  RopeBaselineExperiment RopeOptimizedExperiment
  GemmBaselineExperiment GemmOptimizedExperiment
  GqaAttentionBaselineExperiment GqaAttentionOptimizedExperiment
  RmsNormBaselineExperiment RmsNormOptimizedExperiment
  SwiGluBaselineExperiment SwiGluOptimizedExperiment
)

for project in "${projects[@]}"; do
  mapfile -t tests < <(find "$ROOT/$project/tests" -maxdepth 1 -type f -name '*.py' ! -name '_setup_env.py' | sort)
  [[ ${#tests[@]} -gt 0 ]] || { echo "[FAIL] no tests in $project"; exit 1; }
  for test_file in "${tests[@]}"; do
    echo "[FOUND] ${test_file#$ROOT/}"
    if (( execute )); then
      (
        cd "$ROOT/$project"
        if [[ -x scripts/run_test.sh ]]; then bash scripts/run_test.sh "$test_file"; else python3 "$test_file"; fi
      )
    fi
  done
done

if (( ! execute )); then
  echo '[INFO] discovery only. Pass --execute only on an NPU-ready host; tests requiring a model or torch_npu inherit their original requirements.'
fi
