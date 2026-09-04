#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/build/bin/spmv_acl"
if [[ ! -x "${BIN}" ]]; then bash "${ROOT_DIR}/scripts/build.sh"; fi
exec "${BIN}" "$@"
