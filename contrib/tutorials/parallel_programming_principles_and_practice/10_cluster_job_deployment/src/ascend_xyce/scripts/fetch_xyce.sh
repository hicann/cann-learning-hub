#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XYCE_DIR="${ROOT_DIR}/third_party/Xyce/source"
XYCE_REPO="${XYCE_REPO:-https://github.com/Xyce/Xyce.git}"

if [[ -d "${XYCE_DIR}/.git" ]]; then
    echo "Xyce source already exists: ${XYCE_DIR}"
    exit 0
fi

echo "Cloning Xyce source from ${XYCE_REPO}"
mkdir -p "$(dirname "${XYCE_DIR}")"
git clone --depth 1 "${XYCE_REPO}" "${XYCE_DIR}"
