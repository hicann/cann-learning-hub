#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/output"
AIR_PATH="${OUTPUT_DIR}/single_add.air"
OM_PATH="${OUTPUT_DIR}/single_add.om"
SOC_VERSION="${SOC_VERSION:-Ascend910B4}"

info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }

if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
  error "Please source CANN 9.0.0 set_env.sh first (ASCEND_HOME_PATH is invalid)."
  exit 1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt")"
  if [[ -n "${cached_source}" && "${cached_source}" != "${SCRIPT_DIR}" ]]; then
    cmake -E remove_directory "${BUILD_DIR}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"
jobs="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"

info "Step 1/5: configure and build offline operator project"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${jobs}"

info "Step 2/5: compile Ascend C kernel package for ${SOC_VERSION}"
if ! cmake --build "${BUILD_DIR}" --target binary -j"${jobs}"; then
  # Operator template layouts may expose the concrete per-SoC target instead
  # of the aggregate `binary` target.
  cmake --build "${BUILD_DIR}" --target AddCustom_ascend910b -j"${jobs}"
fi

# A RUN package is normally produced by CMake's `package` target.  Installing
# the generated tree directly is convenient for this self-contained sample;
# if a toolkit does not expose RUN-mode install rules, fall back to the
# official package target and use its CPack staging directory below.
INSTALL_ROOT="${BUILD_DIR}/install"
if ! cmake --install "${BUILD_DIR}" --prefix "${INSTALL_ROOT}" >/dev/null 2>&1; then
  info "RUN-mode install tree is unavailable; will inspect the CPack package staging directory."
fi

CUSTOM_OPP_ROOT=""
for candidate in \
  "${INSTALL_ROOT}/vendors/customize" \
  "${INSTALL_ROOT}/packages/vendors/customize" \
  "${BUILD_DIR}/vendors/customize" \
  "${BUILD_DIR}/packages/vendors/customize"; do
  if [[ -d "${candidate}/op_proto" && -d "${candidate}/op_impl" ]]; then
    CUSTOM_OPP_ROOT="${candidate}"
    break
  fi
done
if [[ -z "${CUSTOM_OPP_ROOT}" ]]; then
  for cpack_dir in "${BUILD_DIR}/_CPack_Packages" "${BUILD_DIR}/_CPack_Package"; do
    package_root="$(find "${cpack_dir}" -type d \
      -path '*/packages/vendors/customize' -print -quit 2>/dev/null || true)"
    if [[ -n "${package_root}" && -d "${package_root}/op_proto" ]]; then
      CUSTOM_OPP_ROOT="${package_root}"
      break
    fi
  done
fi
if [[ -z "${CUSTOM_OPP_ROOT}" ]]; then
  # Keep the error below actionable if a toolkit generated a non-standard
  # install location; the package target provides another supported layout.
  if ! cmake --build "${BUILD_DIR}" --target package -j"${jobs}" >/dev/null; then
    info "The toolkit has no CPack package target; checking for an existing RUN artifact."
  fi
  for cpack_dir in "${BUILD_DIR}/_CPack_Packages" "${BUILD_DIR}/_CPack_Package"; do
    package_root="$(find "${cpack_dir}" -type d \
      -path '*/packages/vendors/customize' -print -quit 2>/dev/null || true)"
    if [[ -n "${package_root}" && -d "${package_root}/op_proto" ]]; then
      CUSTOM_OPP_ROOT="${package_root}"
      break
    fi
  done
fi
if [[ -z "${CUSTOM_OPP_ROOT}" ]]; then
  # The RUN package staging directory may be retained under the generated
  # *.run directory even when `cmake --install` is unavailable.
  run_stage="$(find "${BUILD_DIR}" -type d -name '*.run' -print -quit 2>/dev/null || true)"
  if [[ -n "${run_stage}" ]]; then
    package_root="$(find "${run_stage}" -type d \
      -path '*/packages/vendors/customize' -print -quit 2>/dev/null || true)"
    if [[ -n "${package_root}" && -d "${package_root}/op_proto" ]]; then
      CUSTOM_OPP_ROOT="${package_root}"
    fi
  fi
fi
if [[ -z "${CUSTOM_OPP_ROOT}" ]]; then
  proto_so="$(find "${BUILD_DIR}" -type f -name 'libcust_opsproto*.so' -print -quit 2>/dev/null || true)"
  if [[ -n "${proto_so}" ]]; then
    # Some template layouts leave the library in the build tree instead of
    # installing the complete vendor tree.  Walk upward until both OPP
    # subdirectories are present; do not treat the .so path itself as a root.
    proto_dir="$(dirname "${proto_so}")"
    while [[ "${proto_dir}" != "/" ]]; do
      if [[ -d "${proto_dir}/op_proto" && -d "${proto_dir}/op_impl" ]]; then
        CUSTOM_OPP_ROOT="${proto_dir}"
        break
      fi
      [[ "${proto_dir}" == "${BUILD_DIR}" ]] && break
      proto_dir="$(dirname "${proto_dir}")"
    done
  fi
fi
if [[ -z "${CUSTOM_OPP_ROOT}" || ! -d "${CUSTOM_OPP_ROOT}/op_proto" ]]; then
  error "The installed custom OPP package was not found below ${BUILD_DIR}."
  exit 1
fi

# CANN expects ASCEND_CUSTOM_OPP_PATH to point at the vendor directory that
# contains op_proto/op_tiling/op_impl, not at a single .so file.
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
library_dirs="${CUSTOM_OPP_ROOT}/op_api/lib"
while IFS= read -r shared_object; do
  library_dirs="$(dirname "${shared_object}"):${library_dirs}"
done < <(find "${CUSTOM_OPP_ROOT}" -type f -name '*.so' -print 2>/dev/null)
export LD_LIBRARY_PATH="${library_dirs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
info "ASCEND_CUSTOM_OPP_PATH=${CUSTOM_OPP_ROOT}"

info "Step 3/5: export AIR graph"
rm -f "${AIR_PATH}"
(
  cd "${OUTPUT_DIR}"
  "${BUILD_DIR}/bin/offline_add_graph_build"
)
[[ -s "${AIR_PATH}" ]] || { error "AIR was not generated: ${AIR_PATH}"; exit 1; }

info "Step 4/5: convert AIR to OM with ATC"
rm -f "${OM_PATH}"
ATC_BIN="${ASCEND_HOME_PATH}/bin/atc"
if [[ ! -x "${ATC_BIN}" ]]; then
  ATC_BIN="$(command -v atc || true)"
fi
[[ -x "${ATC_BIN}" ]] || { error "atc was not found"; exit 1; }
"${ATC_BIN}" \
  --model="${AIR_PATH}" \
  --framework=1 \
  --output="${OM_PATH%.om}" \
  --soc_version="${SOC_VERSION}"
[[ -s "${OM_PATH}" ]] || { error "OM was not generated: ${OM_PATH}"; exit 1; }

info "Step 5/5: execute OM with ACL"
"${BUILD_DIR}/bin/offline_add_model_exec" "${OM_PATH}"
info "Offline sample pipeline finished."
