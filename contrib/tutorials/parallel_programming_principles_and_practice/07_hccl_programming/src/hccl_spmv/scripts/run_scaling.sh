#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MATRIX="U1"; TABLE="${RANK_TABLE_FILE:-}"; WARMUP=10; REPEAT=100; NPU_LIST="2,4,8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix) MATRIX="$2"; shift 2;;
    --rank-table) TABLE="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --repeat) REPEAT="$2"; shift 2;;
    --npus-list) NPU_LIST="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
mkdir -p "${ROOT_DIR}/results"
IFS=',' read -r -a npu_counts <<< "${NPU_LIST}"
status=0
for npus in "${npu_counts[@]}"; do
  log="${ROOT_DIR}/results/${MATRIX}_${npus}npu.log"
  echo "===== ${MATRIX}, ${npus} NPU =====" | tee "${log}"
  args=(--npus "${npus}" --matrix "${MATRIX}" --warmup "${WARMUP}" --repeat "${REPEAT}")
  if [[ -n "${TABLE}" ]]; then args+=(--rank-table "${TABLE}"); fi
  if ! bash "${ROOT_DIR}/scripts/run.sh" "${args[@]}" | tee -a "${log}"; then
    echo "FAILED: ${MATRIX}, ${npus} NPU" | tee -a "${log}"
    status=1
  fi
done
exit "${status}"
