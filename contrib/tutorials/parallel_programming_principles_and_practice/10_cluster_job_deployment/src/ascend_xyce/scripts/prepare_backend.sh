#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend/gmres/Ascend-GMRES"
BACKEND_REPO="${ASCEND_GMRES_REPO:-git@gitcode.com:maeveyixue/Ascend-GMRES.git}"

# The course copy vendors the exact dependency used by the notebooks so that
# learners do not need a network clone or an unrelated sibling checkout.
if [[ -f "${BACKEND_DIR}/CMakeLists.txt" ]]; then
    echo "${BACKEND_DIR}"
    exit 0
fi

if [[ -n "${ASCEND_GMRES_DIR:-}" && -f "${ASCEND_GMRES_DIR}/CMakeLists.txt" ]]; then
    echo "${ASCEND_GMRES_DIR}"
    exit 0
fi

if [[ -f "${ROOT_DIR}/../Ascend-GMRES/CMakeLists.txt" ]]; then
    echo "${ROOT_DIR}/../Ascend-GMRES"
    exit 0
fi

if [[ ! -d "${BACKEND_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${BACKEND_DIR}")"
    git clone "${BACKEND_REPO}" "${BACKEND_DIR}"
else
    git -C "${BACKEND_DIR}" pull --ff-only
fi

echo "${BACKEND_DIR}"
