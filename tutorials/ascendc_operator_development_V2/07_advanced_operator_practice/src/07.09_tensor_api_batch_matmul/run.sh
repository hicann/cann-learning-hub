#!/usr/bin/env bash
# CANN 版本要求：不低于 9.2.0（Tensor API asc::te:: 接口自 9.2.0 起提供）
set -euo pipefail

npu_arch="dav-3510"
case_name="practice"

for arg in "$@"; do
    case "$arg" in
        --npu-arch=*)
            npu_arch="${arg#*=}"
            ;;
        --case=*)
            case_name="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: bash run.sh [--npu-arch=dav-3510] [--case=practice|answer]"
            exit 0
            ;;
    esac
done

if [ "$npu_arch" != "dav-3510" ]; then
    echo "Tensor API BatchMatmul sample only supports --npu-arch=dav-3510"
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
    echo "[VERIFY] ${name}"
    python3 scripts/verify_result.py "${out_file}"
}

case "$case_name" in
    practice)
        run_one practice batch_matmul_practice
        ;;
    answer)
        run_one answer batch_matmul_answer
        ;;
    all)
        run_one practice batch_matmul_practice
        run_one answer batch_matmul_answer
        ;;
    *)
        echo "Unsupported case: ${case_name}"
        echo "Use practice, answer, or all."
        exit 1
        ;;
esac
