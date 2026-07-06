#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf "${SCRIPT_DIR}/build"
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build "${SCRIPT_DIR}/build" -j
cd "${SCRIPT_DIR}/build"
export LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH:-}"
python3 ../practice_test.py
