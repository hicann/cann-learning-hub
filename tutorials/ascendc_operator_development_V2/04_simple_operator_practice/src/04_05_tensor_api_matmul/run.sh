#!/usr/bin/env bash
set -euo pipefail

npu_arch="dav-3510"
case_name="all"

for arg in "$@"; do
    case "$arg" in
        --npu-arch=*)
            npu_arch="${arg#*=}"
            ;;
        --case=*)
            case_name="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: bash run.sh [--npu-arch=dav-3510] [--case=single_core|multi_core|double_buffer|practice|all]"
            exit 0
            ;;
    esac
done

if [ "$npu_arch" != "dav-3510" ]; then
    echo "Tensor API matrix sample only supports --npu-arch=dav-3510"
    exit 1
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "Please set ASCEND_HOME_PATH before building."
    exit 1
fi

if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi

rm -rf build_out input output
cmake -S . -B build_out -DCMAKE_ASC_ARCHITECTURES="$npu_arch"
python3 scripts/gen_data.py
mkdir -p output

run_one() {
    local name="$1"
    local binary="$2"
    local out_file="output/${name}.bin"
    echo "[BUILD] ${binary}"
    cmake --build build_out -j"$(nproc)" --target "${binary}"
    echo "[RUN] ${name}"
    ./build_out/${binary}
    python3 scripts/verify_result.py "${out_file}"
}

case "$case_name" in
    single_core)
        run_one single_core tensor_mmad_single_core
        ;;
    multi_core)
        run_one multi_core tensor_mmad_multi_core
        ;;
    double_buffer)
        run_one double_buffer tensor_mmad_double_buffer
        ;;
    practice)
        run_one practice tensor_mmad_practice
        ;;
    all)
        run_one single_core tensor_mmad_single_core
        run_one multi_core tensor_mmad_multi_core
        run_one double_buffer tensor_mmad_double_buffer
        ;;
    *)
        echo "Unsupported case: ${case_name}"
        echo "Use single_core, multi_core, double_buffer, practice, or all."
        exit 1
        ;;
esac
