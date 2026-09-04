#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
MPI_PROCESSES="${MPI_PROCESSES:-16}"

if [[ ! -x "${BUILD_DIR}/spmv_mpi" ]]; then
    bash "${ROOT_DIR}/scripts/build.sh"
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "error: mpirun is not installed or not in PATH" >&2
    exit 1
fi

mkdir -p "${ROOT_DIR}/results"
cd "${ROOT_DIR}"
mpirun -np "${MPI_PROCESSES}" "${BUILD_DIR}/spmv_mpi" "$@" \
    2>&1 | tee "${ROOT_DIR}/results/mpi_run.log"
