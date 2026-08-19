#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/setup_cannlab_env.sh"

fail() { echo "[FAIL] $*" >&2; exit 1; }
check_command() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

[[ "$(uname -m)" == "aarch64" ]] || fail "expected aarch64, got $(uname -m)"
[[ -d "$ASCEND_HOME_PATH" ]] || fail "CANN directory not found: $ASCEND_HOME_PATH"
[[ -d "$QWEN_OPS_MODEL_PATH" ]] || fail "model directory not found: $QWEN_OPS_MODEL_PATH"
[[ -f "$QWEN_OPS_MODEL_PATH/model.safetensors" ]] || fail "model.safetensors is missing"

for command_name in cmake make gcc g++ npu-smi "$PYTHON_BIN"; do
    check_command "$command_name"
done

npu-smi info
"$PYTHON_BIN" - <<'PY'
import sys
import torch
import torch_npu
import transformers
import einops

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("transformers:", transformers.__version__)
print("einops:", einops.__version__)
if not hasattr(torch, "npu") or not torch.npu.is_available():
    raise SystemExit("torch_npu cannot access an NPU")
print("NPU available: yes")
PY

echo "[PASS] CannLearningLab environment is ready"
