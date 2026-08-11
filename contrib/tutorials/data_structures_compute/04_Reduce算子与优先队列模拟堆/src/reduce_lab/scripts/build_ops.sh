#!/bin/bash
set -euo pipefail

# 目标平台：TARGET=ascend310b 或 TARGET=ascend910b
TARGET="${TARGET:-ascend910b}"
echo "=== Building for target: ${TARGET} ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/custom_ops/src"
JSON_FILE="$PROJECT_DIR/custom_ops/json/reduce_lite_ops.json"
GEN_DIR="$PROJECT_DIR/custom_ops/generated"
OUT_DIR="$GEN_DIR/ReduceLiteOps"

ASCEND_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
MSOPGEN="${ASCEND_PATH}/python/site-packages/bin/msopgen"
if [ ! -x "${MSOPGEN}" ]; then
  MSOPGEN="${ASCEND_PATH}/bin/msopgen"
fi
if [ ! -x "${MSOPGEN}" ]; then
  MSOPGEN="$(command -v msopgen || true)"
fi
if [ -z "${MSOPGEN}" ] || [ ! -x "${MSOPGEN}" ]; then
  echo "msopgen not found. Please run: source /home/developer/Ascend/cann-9.0.0/set_env.sh  (or source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh)" >&2
  exit 1
fi

resolve_project_dir() {
  local d="$1"
  if [ -d "$d/op_host" ] && [ -d "$d/op_kernel" ]; then
    echo "$d"
    return 0
  fi
  if [ -d "$d/ReduceLiteOps/op_host" ] && [ -d "$d/ReduceLiteOps/op_kernel" ]; then
    echo "$d/ReduceLiteOps"
    return 0
  fi
  echo "$d"
}

copy_impls() {
  local dst="$1"
  echo "=== Copy implementation sources ==="

  # ReduceSumLite
  cp -v "$SRC_DIR/ReduceSumLite/op_host/reduce_sum_lite.cpp" "$dst/op_host/reduce_sum_lite.cpp"
  cp -v "$SRC_DIR/ReduceSumLite/op_host/reduce_sum_lite_tiling.h" "$dst/op_host/reduce_sum_lite_tiling.h"
  cp -v "$SRC_DIR/ReduceSumLite/op_kernel/reduce_sum_lite.cpp" "$dst/op_kernel/reduce_sum_lite.cpp"
  cp -v "$SRC_DIR/ReduceSumLite/op_kernel/reduce_sum_lite_tiling.h" "$dst/op_kernel/reduce_sum_lite_tiling.h"

  # ReduceMaxLite
  cp -v "$SRC_DIR/ReduceMaxLite/op_host/reduce_max_lite.cpp" "$dst/op_host/reduce_max_lite.cpp"
  cp -v "$SRC_DIR/ReduceMaxLite/op_host/reduce_max_lite_tiling.h" "$dst/op_host/reduce_max_lite_tiling.h"
  cp -v "$SRC_DIR/ReduceMaxLite/op_kernel/reduce_max_lite.cpp" "$dst/op_kernel/reduce_max_lite.cpp"
  cp -v "$SRC_DIR/ReduceMaxLite/op_kernel/reduce_max_lite_tiling.h" "$dst/op_kernel/reduce_max_lite_tiling.h"

  # TopKReduceLite
  cp -v "$SRC_DIR/TopKReduceLite/op_host/top_k_reduce_lite.cpp" "$dst/op_host/top_k_reduce_lite.cpp"
  cp -v "$SRC_DIR/TopKReduceLite/op_host/top_k_reduce_lite_tiling.h" "$dst/op_host/top_k_reduce_lite_tiling.h"
  cp -v "$SRC_DIR/TopKReduceLite/op_kernel/top_k_reduce_lite.cpp" "$dst/op_kernel/top_k_reduce_lite.cpp"
  cp -v "$SRC_DIR/TopKReduceLite/op_kernel/top_k_reduce_lite_tiling.h" "$dst/op_kernel/top_k_reduce_lite_tiling.h"
}

echo "=== Step 1: Generate msopgen project for 3 operators ==="
mkdir -p "$GEN_DIR"
rm -rf "$OUT_DIR"

ops=(ReduceSumLite ReduceMaxLite TopKReduceLite)

for i in "${!ops[@]}"; do
  op="${ops[$i]}"
  step=$((i + 1))
  total=${#ops[@]}
  if [ "$i" -eq 0 ]; then
    echo "[$step/$total] Generate ${op}"
    "${MSOPGEN}" gen -i "${JSON_FILE}" -op "${op}" -c "ai_core-${TARGET}" -lan cpp -out "${OUT_DIR}"
  else
    OUT_DIR="$(resolve_project_dir "$OUT_DIR")"
    echo "[$step/$total] Append ${op}"
    "${MSOPGEN}" gen -i "${JSON_FILE}" -op "${op}" -c "ai_core-${TARGET}" -lan cpp -m 1 -out "${OUT_DIR}"
  fi
  OUT_DIR="$(resolve_project_dir "$OUT_DIR")"
done

if [ ! -d "$OUT_DIR/op_host" ] || [ ! -d "$OUT_DIR/op_kernel" ]; then
  echo "Generated project is incomplete: $OUT_DIR" >&2
  exit 1
fi

copy_impls "$OUT_DIR"

# Patch BLOCK_DIM based on target platform
case "$TARGET" in
  ascend310b) BDIM=8 ;;
  ascend910b) BDIM=20 ;;
  ascend910b_*) BDIM=24 ;;
  *) BDIM=24 ;;
esac
echo "=== Patching BLOCK_DIM = ${BDIM} for ${TARGET} ==="
sed -i "s/static constexpr uint32_t BLOCK_DIM = 8/static constexpr uint32_t BLOCK_DIM = ${BDIM}/g" "$OUT_DIR/op_host/reduce_sum_lite.cpp"
sed -i "s/static constexpr uint32_t BLOCK_DIM = 8/static constexpr uint32_t BLOCK_DIM = ${BDIM}/g" "$OUT_DIR/op_host/reduce_max_lite.cpp"
sed -i "s/static constexpr uint32_t BLOCK_DIM = 8/static constexpr uint32_t BLOCK_DIM = ${BDIM}/g" "$OUT_DIR/op_host/top_k_reduce_lite.cpp"

if [ -f "$OUT_DIR/CMakePresets.json" ]; then
  if [ -f "$PROJECT_DIR/scripts/patch_cmakepresets.py" ]; then
    python3 "$PROJECT_DIR/scripts/patch_cmakepresets.py" "$OUT_DIR/CMakePresets.json" "${TARGET}"
  fi
fi


# Fix duplicate sections in .ini file (msopgen -m 1 may create duplicates)
INI_FILE="$OUT_DIR/build_out/autogen/aic-${TARGET}-ops-info.ini"
if [ -f "$INI_FILE" ]; then
  python3 -c "
import re, sys
path = sys.argv[1]
with open(path) as f: lines = f.readlines()
seen = set(); out = []; skip = False
for line in lines:
    m = re.match(r'^\[([^\]]+)\]', line)
    if m:
        sec = m.group(1)
        if sec in seen: skip = True; continue
        seen.add(sec); skip = False
    if not skip: out.append(line)
with open(path, 'w') as f: f.writelines(out)
print(f'Deduped .ini: {len(lines)} -> {len(out)} lines')
" "$INI_FILE"
fi

echo "=== Step 2: Build custom operators ==="
cd "$OUT_DIR"
rm -rf build_out
mkdir -p build_out
source "${ASCEND_PATH}/set_env.sh" 2>/dev/null || true
cmake -S . -B build_out --preset=default 2>&1
cmake --build build_out --target binary -j1 2>&1
cmake --build build_out --target package -j1 2>&1

echo "=== Step 3: Install custom OPP package ==="
RUN_PKG="$(find build_out -maxdepth 1 -name 'custom_opp_*.run' | head -1 || true)"
if [ -z "${RUN_PKG}" ]; then
  echo "No custom_opp_*.run package found under $OUT_DIR/build_out" >&2
  exit 1
fi

OLD_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
unset ASCEND_CUSTOM_OPP_PATH
bash "${RUN_PKG}"
if [ -n "${OLD_CUSTOM_OPP_PATH}" ]; then
  export ASCEND_CUSTOM_OPP_PATH="${OLD_CUSTOM_OPP_PATH}"
fi

cd "$PROJECT_DIR"
echo "=== Done ==="
echo "Generated project: $OUT_DIR"
echo "Recommended runtime environment:"
echo "  source ${PROJECT_DIR}/scripts/env_custom_opp.sh"
