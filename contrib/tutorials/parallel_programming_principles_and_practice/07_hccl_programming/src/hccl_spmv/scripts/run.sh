#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/build/bin/hccl_spmv"
if [[ ! -x "${BIN}" ]]; then bash "${ROOT_DIR}/scripts/build.sh"; fi
NPUS="${HCCL_NPUS:-1}"; TABLE="${RANK_TABLE_FILE:-}"; EXTRA=()
mkdir -p "${ROOT_DIR}/results"
while [[ $# -gt 0 ]]; do case "$1" in --npus) NPUS="$2"; shift 2;; --rank-table) TABLE="$2"; shift 2;; *) EXTRA+=("$1"); shift;; esac; done
if [[ "$NPUS" -le 1 ]]; then
  if [[ -n "$TABLE" ]]; then exec env RANK_ID=0 RANK_SIZE=1 DEVICE_ID="${ASCEND_DEVICE_ID:-0}" "${BIN}" "${EXTRA[@]}" --rank-table "$TABLE"; fi
  exec env RANK_ID=0 RANK_SIZE=1 DEVICE_ID="${ASCEND_DEVICE_ID:-0}" "${BIN}" "${EXTRA[@]}"
fi
ROOT_INFO_FILE="/tmp/hccl_spmv_root_info_${PPID}.bin"
rm -f "${ROOT_INFO_FILE}"
PIDS=(); for ((r=0;r<NPUS;r++)); do
  args=(--rank "$r" --world-size "$NPUS" --device "$r" "${EXTRA[@]}")
  if [[ -n "$TABLE" ]]; then args+=(--rank-table "$TABLE"); fi
  RANK_ID="$r" RANK_SIZE="$NPUS" DEVICE_ID="$r" RANK_TABLE_FILE="$TABLE" HCCL_SPMV_ROOT_INFO_FILE="$ROOT_INFO_FILE" "${BIN}" "${args[@]}" >"${ROOT_DIR}/results/rank_${r}.log" 2>&1 & PIDS+=("$!")
done
status=0; for p in "${PIDS[@]}"; do wait "$p" || status=$?; done
# Every case keeps the per-rank logs (results/rank_N.log) and the aggregated
# console output below shows each rank's full log in rank order, so rank 1+
# evidence is never hidden behind rank 0's summary.
for ((r=0;r<NPUS;r++)); do
  echo "===== rank ${r} ====="
  cat "${ROOT_DIR}/results/rank_${r}.log"
done
exit "$status"
