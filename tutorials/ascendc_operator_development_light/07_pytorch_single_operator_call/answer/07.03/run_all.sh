#!/usr/bin/env bash
set -eo pipefail

answer_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_env=${ASCEND_SET_ENV:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann}/set_env.sh}
if [[ ! -f "$cann_env" ]]; then
    echo "CANN environment script not found: $cann_env" >&2
    exit 1
fi
source "$cann_env"
python_bin=${PYTHON_BIN:-$(command -v python3)}
build_root=${BUILD_ROOT:-$answer_root/build}
mkdir -p "$build_root"
build_root=$(cd -- "$build_root" && pwd)

for op in softmax matmul; do
    for binding in pybind torch_library; do
        build_dir="$build_root/$op/$binding"
        cmake -S "$answer_root/$op/$binding" -B "$build_dir" \
            -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
            -DPython3_EXECUTABLE="$python_bin"
        cmake --build "$build_dir" --parallel "${BUILD_JOBS:-2}"
        "$python_bin" "$answer_root/$op/$binding/${op}_test.py" --build-dir "$build_dir"
    done
done
