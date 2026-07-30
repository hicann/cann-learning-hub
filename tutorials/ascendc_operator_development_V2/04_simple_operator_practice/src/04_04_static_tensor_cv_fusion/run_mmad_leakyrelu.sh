#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NPU_ARCH="dav-2201"
COMPILE_ONLY=0
FORCE_RUN=0
RUN_MODE=""

for arg in "$@"; do
  case "${arg}" in
    --npu-arch=*) NPU_ARCH="${arg#*=}" ;;
    --mode=*) RUN_MODE="${arg#*=}" ;;
    --compile-only) COMPILE_ONLY=1 ;;
    --force-run) FORCE_RUN=1 ;;
    -h|--help)
      echo "用法: bash run_mmad_leakyrelu.sh [--mode=sim|npu] [--npu-arch=dav-2201|dav-3510] [--compile-only] [--force-run]"
      exit 0
      ;;
    *)
      echo "未知参数: ${arg}"
      echo "用法: bash run_mmad_leakyrelu.sh [--mode=sim|npu] [--npu-arch=dav-2201|dav-3510] [--compile-only] [--force-run]"
      exit 1
      ;;
  esac
done

case "${NPU_ARCH}" in
  dav-2201|dav-3510) ;;
  *)
    echo "错误：不支持的架构名 '${NPU_ARCH}'，仅支持 dav-2201 或 dav-3510。"
    exit 1
    ;;
esac

case "${RUN_MODE}" in
  "") ;;
  sim|npu) ;;
  *)
    echo "错误：不支持的运行模式 '${RUN_MODE}'，仅支持 sim 或 npu。"
    exit 1
    ;;
esac

resolve_ascend_env() {
  local candidates=()
  if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    candidates+=("${ASCEND_HOME_PATH}")
  fi
  if [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
    candidates+=("${ASCEND_TOOLKIT_HOME}")
  fi

  if [ "$(id -u)" -eq 0 ]; then
    candidates+=("/usr/local/Ascend/cann" "/usr/local/Ascend/ascend-toolkit/latest")
  else
    candidates+=("${HOME}/Ascend/cann" "${HOME}/Ascend/ascend-toolkit/latest")
  fi

  local candidate normalized
  for candidate in "${candidates[@]}"; do
    [ -n "${candidate}" ] || continue
    normalized="${candidate%/}"
    case "$(basename "${normalized}")" in
      x86_64-linux|aarch64-linux) normalized="$(dirname "${normalized}")" ;;
    esac
    if [ -f "${normalized}/set_env.sh" ]; then
      export ASCEND_TOOLKIT_HOME="${normalized}"
      set +u
      source "${normalized}/set_env.sh"
      set -u
      return 0
    fi
  done
  return 1
}

has_npu_device() {
  if command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1; then
    return 0
  fi
  if [ -e /dev/davinci0 ] || [ -e /dev/davinci_manager ]; then
    return 0
  fi
  return 1
}

if ! resolve_ascend_env; then
  echo "错误：未找到 CANN 包。请安装 CANN 包，并执行 source <cann>/set_env.sh 后重新执行。"
  exit 1
fi

export SAMPLE_DEVICE_ID="${SAMPLE_DEVICE_ID:-0}"

if [ -z "${RUN_MODE}" ]; then
  if has_npu_device; then
    RUN_MODE="npu"
  else
    RUN_MODE="sim"
  fi
fi

rm -rf build input output
mkdir -p build
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES="${NPU_ARCH}" -DCMAKE_ASC_RUN_MODE="${RUN_MODE}"
cmake --build build --target mmad_leakyrelu_demo -j"$(nproc)"

if [ "${COMPILE_ONLY}" -eq 1 ]; then
  echo "compile pass: mmad_leakyrelu_demo"
  exit 0
fi

case "${RUN_MODE}" in
  sim)
    echo "当前运行模式: sim。将执行 sim 模式运行与精度校验。"
    ;;
  npu)
    if [ "${FORCE_RUN}" -eq 0 ] && ! has_npu_device; then
      echo "错误：当前为 npu 模式，但未在 /dev 下检测到 NPU 设备节点。请改用 --mode=sim，或在具备 NPU 的环境执行。"
      exit 1
    fi
    echo "当前运行模式: npu。将执行 NPU 运行与精度校验。"
    ;;
esac

python3 scripts/gen_data.py --sample=mmad_leakyrelu
timeout "${SAMPLE_RUN_TIMEOUT:-120}" ./build/mmad_leakyrelu_demo
python3 scripts/verify_result.py output/output.bin output/golden.bin
