#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/build/bin/dis_gmres"
NPUS="${HCCL_NPUS:-1}"
RANK_TABLE="${RANK_TABLE_FILE:-}"
RUN_BASELINE=1
TOTAL_THREADS="${DIS_GMRES_TOTAL_THREADS:-}"
THREADS_PER_RANK="${DIS_GMRES_THREADS_PER_RANK:-16}"
HOST_CPUS="${DIS_GMRES_HOST_CPUS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf 16)}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npus) NPUS="$2"; shift 2 ;;
    --rank-table) RANK_TABLE="$2"; shift 2 ;;
    --no-baseline) RUN_BASELINE=0; shift ;;
    --total-threads) TOTAL_THREADS="$2"; shift 2 ;;
    --threads-per-rank) THREADS_PER_RANK="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -x "${BIN}" ]]; then
  bash "${ROOT_DIR}/scripts/build.sh"
fi
if ! [[ "${NPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--npus must be a positive integer" >&2
  exit 2
fi
if [[ -n "${TOTAL_THREADS}" ]] && ! [[ "${TOTAL_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--total-threads must be a positive integer when specified" >&2
  exit 2
fi
if ! [[ "${THREADS_PER_RANK}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${HOST_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "threads-per-rank and detected host CPU count must be positive integers" >&2
  exit 2
fi

# Cap every rank instead of letting every process consume all host CPUs.  The
# default scales CPU helpers with ranks up to the host capacity.  Setting
# DIS_GMRES_TOTAL_THREADS switches to a fixed aggregate budget for controlled
# resource-equality experiments.
if [[ -n "${TOTAL_THREADS}" ]]; then
  EFFECTIVE_TOTAL="${TOTAL_THREADS}"
  ((EFFECTIVE_TOTAL > HOST_CPUS)) && EFFECTIVE_TOTAL="${HOST_CPUS}"
  BASELINE_THREADS="${EFFECTIVE_TOTAL}"
  RANK_THREADS=$(((EFFECTIVE_TOTAL + NPUS - 1) / NPUS))
  THREAD_MODE="fixed-total"
else
  BASELINE_THREADS="${THREADS_PER_RANK}"
  ((BASELINE_THREADS > HOST_CPUS)) && BASELINE_THREADS="${HOST_CPUS}"
  RANK_CAPACITY=$((HOST_CPUS / NPUS))
  ((RANK_CAPACITY < 1)) && RANK_CAPACITY=1
  RANK_THREADS="${THREADS_PER_RANK}"
  ((RANK_THREADS > RANK_CAPACITY)) && RANK_THREADS="${RANK_CAPACITY}"
  THREAD_MODE="capped-per-rank"
fi
OMP_ENV=(OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0)
echo "OpenMP budget: mode=${THREAD_MODE}, host=${HOST_CPUS}, baseline=${BASELINE_THREADS}, per-rank=${RANK_THREADS}, ranks=${NPUS}"

mkdir -p "${ROOT_DIR}/results"
BASELINE_MS=""
if [[ "${NPUS}" -gt 1 && "${RUN_BASELINE}" == "1" ]]; then
  BASELINE_LOG="${ROOT_DIR}/results/single_baseline.log"
  env "${OMP_ENV[@]}" OMP_NUM_THREADS="${BASELINE_THREADS}" \
    RANK_ID=0 RANK_SIZE=1 DEVICE_ID="${ASCEND_DEVICE_ID:-0}" RANK_TABLE_FILE= \
    "${BIN}" "${EXTRA[@]}" >"${BASELINE_LOG}" 2>&1
  BASELINE_MS="$(awk -F= '/^RESULT_TOTAL_MS=/{value=$2} END{print value}' "${BASELINE_LOG}")"
  if [[ -z "${BASELINE_MS}" ]]; then
    cat "${BASELINE_LOG}" >&2
    echo "failed to parse single-device baseline" >&2
    exit 2
  fi
fi

ROOT_INFO_FILE="/tmp/dis_gmres_root_info_${$}.bin"
cleanup() {
  rm -f "${ROOT_INFO_FILE}"
}
trap cleanup EXIT
rm -f "${ROOT_INFO_FILE}"

PIDS=()
for ((rank=0; rank<NPUS; ++rank)); do
  ARGS=(--rank "${rank}" --world-size "${NPUS}" --device "${rank}" "${EXTRA[@]}")
  if [[ -n "${RANK_TABLE}" ]]; then
    ARGS+=(--rank-table "${RANK_TABLE}")
  fi
  if [[ -n "${BASELINE_MS}" ]]; then
    ARGS+=(--single-baseline-ms "${BASELINE_MS}")
  fi
  env "${OMP_ENV[@]}" OMP_NUM_THREADS="${RANK_THREADS}" \
    RANK_ID="${rank}" RANK_SIZE="${NPUS}" DEVICE_ID="${rank}" \
    RANK_TABLE_FILE="${RANK_TABLE}" DIS_GMRES_ROOT_INFO_FILE="${ROOT_INFO_FILE}" \
    "${BIN}" "${ARGS[@]}" >"${ROOT_DIR}/results/rank_${rank}.log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || STATUS=$?
done
cat "${ROOT_DIR}/results/rank_0.log"
if [[ "${STATUS}" -ne 0 ]]; then
  echo "one or more ranks failed; inspect results/rank_*.log" >&2
fi
exit "${STATUS}"
