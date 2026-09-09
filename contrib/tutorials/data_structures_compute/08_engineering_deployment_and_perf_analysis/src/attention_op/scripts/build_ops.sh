#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPS_DIR="$LAB_DIR/custom_ops/generated/AttentionCustom"

echo "=== Building AttentionCustom operator ==="
echo "OPS_DIR: $OPS_DIR"

# ASC cmake 布局探测：兼容 CANN 9.0.0（compiler/tikcpp）与 8.5.0（aarch64-linux/tikcpp）
# 在线 Notebook 环境通常缺少 ASC_DIR / 未将 tikcpp 目录加入 CMAKE_PREFIX_PATH，
# 此处自动补全，保证 find_package(ASC REQUIRED) 可用
ASC_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
ASC_CMAKE_DIR=""
for sub in compiler/tikcpp/ascendc_kernel_cmake \
           aarch64-linux/tikcpp/ascendc_kernel_cmake \
           arm64-linux/tikcpp/ascendc_kernel_cmake; do
    cand="$ASC_HOME/$sub"
    if [ -f "$cand/ASCConfig.cmake" ] || [ -f "$cand/asc-config.cmake" ]; then
        ASC_CMAKE_DIR="$cand"
        break
    fi
done
if [ -n "$ASC_CMAKE_DIR" ]; then
    export ASC_DIR="$ASC_CMAKE_DIR"
    case ":${CMAKE_PREFIX_PATH:-}:" in
        *":$ASC_CMAKE_DIR"*) : ;;
        *) export CMAKE_PREFIX_PATH="$ASC_CMAKE_DIR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" ;;
    esac
    echo "ASC_DIR=$ASC_CMAKE_DIR"
fi

cd "$OPS_DIR"

# 适配 CMakePresets.json 的 CANN 安装路径为当前 $ASCEND_HOME_PATH
if [ -f CMakePresets.json ]; then
    python3 "$SCRIPT_DIR/patch_cmakepresets.py" CMakePresets.json ascend910b
fi

# Clean previous build
rm -rf build_out

# Run the build（直接 cmake --preset，避免依赖生成工程 build.sh 的 preset_parse.py，
# 该工具路径随 CANN 版本布局变化）
cmake -S . -B build_out --preset=default
cmake --build build_out --target binary -j$(nproc)
cmake --build build_out --target package -j$(nproc)

echo ""
echo "=== Build completed ==="
ls -la build_out/

# Find and install the .run package
RUN_FILE=$(find build_out -maxdepth 1 -name "custom_opp_*.run" -type f | head -1)
if [ -z "$RUN_FILE" ]; then
    echo "ERROR: No .run file found in build_out"
    exit 1
fi

echo ""
echo "=== Installing operator package to \${HOME}/vendors/customize ==="
chmod +x "$RUN_FILE"
bash "$RUN_FILE" --install-path="${HOME}" --quiet

echo "=== Operator package installed ==="
