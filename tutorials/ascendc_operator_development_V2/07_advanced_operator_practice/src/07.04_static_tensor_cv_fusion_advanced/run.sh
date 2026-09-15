#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

resolve_cann() {
  local candidate
  for candidate in "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" \
      /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit/latest; do
    [ -n "${candidate}" ] || continue
    if [ -f "${candidate%/}/set_env.sh" ]; then
      set +u
      source "${candidate%/}/set_env.sh"
      set -u
      return 0
    fi
  done
  return 1
}

resolve_cann || {
  echo "错误：未找到 CANN，请先设置 ASCEND_HOME_PATH 或 ASCEND_TOOLKIT_HOME。"
  exit 1
}

rm -rf build input output
mkdir -p build output
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DCMAKE_ASC_RUN_MODE=npu
cmake --build build -j"$(nproc)"
python3 scripts/gen_data.py

./build/mmad_gelu_adv
python3 scripts/verify_result.py output/output.bin output/golden.bin
