#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROPE_BUILD_DIR:-$SRC_DIR/../build}"
CANN_PREFIX="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}}"
: "${CANN_PREFIX:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}"
source "$CANN_PREFIX/set_env.sh"
mkdir -p "$BUILD_DIR"

# Compile the RTC Host; the Device Kernel is compiled at runtime.
"${CXX:-g++}" -std=c++17 \
  -I"$CANN_PREFIX/include" -I"$CANN_PREFIX/include/acl" \
  -L"$CANN_PREFIX/lib64" -Wl,-rpath,"$CANN_PREFIX/lib64" \
  "$SRC_DIR/rope_simd_rtc.cpp" -lascendcl -lacl_rtc \
  -o "$BUILD_DIR/rope_simd_rtc"

LD_LIBRARY_PATH="$CANN_PREFIX/lib64:${LD_LIBRARY_PATH:-}" \
  "$BUILD_DIR/rope_simd_rtc" --kernel "$SRC_DIR/rope_simd_kernel.cpp" "$@"
