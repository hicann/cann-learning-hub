#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NPU_LIST="${NPU_LIST:-1,2,4,8}"
MATRIX="${MATRIX:-U2}"
SUMMARY="${ROOT_DIR}/results/scaling_${MATRIX}.csv"
IFS=',' read -r -a COUNTS <<< "${NPU_LIST}"
mkdir -p "${ROOT_DIR}/results"
echo "npus,total_ms,residual" >"${SUMMARY}"

for count in "${COUNTS[@]}"; do
  LOG="${ROOT_DIR}/results/scaling_${MATRIX}_${count}.log"
  bash "${ROOT_DIR}/scripts/run.sh" --npus "${count}" --matrix "${MATRIX}" \
    --warmup "${WARMUP:-0}" --repeat "${REPEAT:-3}" "$@" | tee "${LOG}"
  TOTAL="$(awk -F= '/^RESULT_TOTAL_MS=/{value=$2} END{print value}' "${LOG}")"
  RESIDUAL="$(awk -F= '/^RESULT_RESIDUAL=/{value=$2} END{print value}' "${LOG}")"
  echo "${count},${TOTAL},${RESIDUAL}" >>"${SUMMARY}"
done
echo "saved ${SUMMARY}"
