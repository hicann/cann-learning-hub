#!/usr/bin/env bash
set -eo pipefail

if [[ -n "${ASCEND_SET_ENV:-}" && -f "${ASCEND_SET_ENV}" ]]; then
    source "${ASCEND_SET_ENV}"
elif [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
else
    source /usr/local/Ascend/cann/set_env.sh
fi

python3 - <<'PY'
import torch
import torch_npu
import pybind11

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("pybind11:", pybind11.__version__)
print("NPU可用数量:", torch.npu.device_count())
PY
