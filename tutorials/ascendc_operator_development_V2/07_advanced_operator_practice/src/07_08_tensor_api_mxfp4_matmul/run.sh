#!/usr/bin/env bash
# CANN 版本要求：不低于 9.2.0（Tensor API asc::te:: 接口自 9.2.0 起提供）
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
            echo "Usage: bash run.sh [--npu-arch=dav-3510] [--case=high_performance|quant_practice|quant_answer|all]"
            exit 0
            ;;
    esac
done

if [ "$npu_arch" != "dav-3510" ]; then
    echo "Tensor API MxFP4 sample only supports --npu-arch=dav-3510"
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
mkdir -p output

run_one() {
    local name="$1"
    local binary="$2"
    local gen_script="$3"
    local verify_script="$4"
    local out_file="output/${name}.bin"
    echo "[GEN] ${gen_script}"
    python3 "scripts/${gen_script}"
    echo "[BUILD] ${binary}"
    cmake --build build_out -j"$(nproc)" --target "${binary}"
    echo "[RUN] ${name}"
    ./build_out/${binary}
    echo "[VERIFY] ${name}"
    python3 "scripts/${verify_script}" "${out_file}"
}

case "$case_name" in
    high_performance)
        run_one high_performance mmad_mx_high_performance gen_data.py verify_result.py
        ;;
    quant_practice)
        run_one quant_practice mmad_mx_quant_practice gen_data_quant.py verify_result_quant.py
        ;;
    quant_answer)
        run_one quant_answer mmad_mx_quant_answer gen_data_quant.py verify_result_quant.py
        ;;
    all)
        run_one high_performance mmad_mx_high_performance gen_data.py verify_result.py
        ;;
    *)
        echo "Unsupported case: ${case_name}"
        echo "Use high_performance, quant_practice, quant_answer, or all."
        exit 1
        ;;
esac
