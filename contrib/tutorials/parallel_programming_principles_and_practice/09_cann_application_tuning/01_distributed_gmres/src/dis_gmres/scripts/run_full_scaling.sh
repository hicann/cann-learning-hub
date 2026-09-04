#!/usr/bin/env bash
set -euo pipefail

# Run every reference matrix on 2/4/8 ranks and emit both machine-readable CSV
# and a Markdown table suitable for the README.  A rank table must describe the
# exact communicator size; use --rank-table-dir with rank_table_2p.json,
# rank_table_4p.json and rank_table_8p.json when using HcclCommInitClusterInfo.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/build/bin/dis_gmres"
MATRIX_LIST="${MATRIX_LIST:-U1,U2,L1,L2,B1,B2}"
NPU_LIST="${NPU_LIST:-2,4,8}"
WARMUP="${WARMUP:-0}"
REPEAT="${REPEAT:-10}"
TABLE_DIR="${RANK_TABLE_DIR:-}"
CSV="${FULL_SCALING_CSV:-${ROOT_DIR}/results/dis_gmres_scaling.csv}"
MARKDOWN="${FULL_SCALING_MD:-${ROOT_DIR}/results/dis_gmres_scaling.md}"
LOG_DIR="${ROOT_DIR}/results/dis_gmres_scaling_logs"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrices|--matrix-list) MATRIX_LIST="$2"; shift 2 ;;
    --npus-list) NPU_LIST="$2"; shift 2 ;;
    --rank-table-dir) TABLE_DIR="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --markdown) MARKDOWN="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# --- CSV parsing / validation helpers (also used by --selfcheck, which must
# run before any build or rank-table setup) -----------------------------------

value_line() {
  local log="$1"
  local label="$2"
  awk -v label="${label}" '$0 ~ "^" label " = " {sub("^" label " = ", ""); print $1; exit}' "${log}"
}

section_total() {
  local log="$1"
  local section="$2"
  # Prefix match so both "Distributed NPU:" and "Distributed Host Stub:" work.
  awk -v section="${section}" '$0 ~ "^" section {getline; print $4; exit}' "${log}"
}

section_speedup() {
  local log="$1"
  awk '/^Speedup:/{getline; gsub(/x/, "", $1); print $1; exit}' "${log}"
}

# Strict decimal/scientific number check: optional sign, one integer/fraction
# part, and an optional exponent with sign and digits. Rejects "e", "..", "1e",
# "1.2.3", "" and any non-numeric payload.
is_number() {
  local value="$1"
  [[ "$value" =~ ^[+-]?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?$ ]]
}

write_case() {
  local matrix="$1" npus="$2" log="$3" status="$4" baseline="$5"
  local rows cols nnz iterations residual single distributed speedup
  local spmv dot axpy norm hccl transfer sync allreduce allgather error backend
  local threads omp_min vector_ops
  rows="$(awk -F'= ' '/^rows =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  cols="$(awk -F'= ' '/^cols =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  nnz="$(awk -F'= ' '/^nnz =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  iterations="$(awk -F'= ' '/^iteration =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  residual="$(awk -F= '/^RESULT_RESIDUAL=/{print $2; exit}' "${log}" 2>/dev/null || true)"
  # Reuse the already-parsed single-rank baseline (RESULT_TOTAL_MS of the 1-rank
  # run) instead of searching for a label the program never printed.
  single="${baseline}"
  distributed="$(section_total "${log}" 'Distributed' 2>/dev/null || true)"
  speedup="$(section_speedup "${log}" 2>/dev/null || true)"
  spmv="$(value_line "${log}" 'SpMV' 2>/dev/null || true)"
  dot="$(value_line "${log}" 'Dot' 2>/dev/null || true)"
  axpy="$(value_line "${log}" 'AXPY' 2>/dev/null || true)"
  norm="$(value_line "${log}" 'Norm' 2>/dev/null || true)"
  hccl="$(value_line "${log}" 'HCCL communication' 2>/dev/null || true)"
  transfer="$(value_line "${log}" 'ACL transfer' 2>/dev/null || true)"
  sync="$(value_line "${log}" 'synchronization' 2>/dev/null || true)"
  allreduce="$(value_line "${log}" 'AllReduce calls' 2>/dev/null || true)"
  allgather="$(value_line "${log}" 'AllGather calls' 2>/dev/null || true)"
  error="$(awk -F'= ' '/^solution relative error =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  backend="$(awk -F'= ' '/^backend =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  threads="$(awk -F'= ' '/^OpenMP threads per rank =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  omp_min="$(awk -F'= ' '/^OpenMP minimum elements =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  vector_ops="$(awk -F'= ' '/^vector operations =/{print $2; exit}' "${log}" 2>/dev/null || true)"
  # No fake pass: a case is only pass when every program-provided schema field
  # is present and strictly well formed (numeric where numeric, exact backend
  # label, non-empty vector_ops). Missing or malformed fields downgrade the
  # case to fail and the caller exits nonzero.
  if [[ "${status}" == "pass" ]]; then
    local field
    local numeric_fields=(rows cols nnz iterations residual single distributed speedup \
                          spmv dot axpy norm hccl transfer sync allreduce allgather \
                          error threads omp_min)
    for field in "${numeric_fields[@]}"; do
      if [[ -z "${!field}" ]] || ! is_number "${!field}"; then
        status="fail"
        echo "VALIDATION FAILED: ${matrix}, ${npus} NPU; field '${field}' missing or malformed (value='${!field}'); see ${log}" >&2
        break
      fi
    done
    if [[ "${status}" == "pass" && -z "${vector_ops}" ]]; then
      status="fail"
      echo "VALIDATION FAILED: ${matrix}, ${npus} NPU; field 'vector_ops' missing; see ${log}" >&2
    fi
    if [[ "${status}" == "pass" && "${backend}" != "ACL + HCCL" ]]; then
      status="fail"
      echo "VALIDATION FAILED: ${matrix}, ${npus} NPU; backend must be exactly 'ACL + HCCL' (got '${backend}'); see ${log}" >&2
    fi
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${matrix}" "${npus}" "${rows}" "${cols}" "${nnz}" "${iterations}" "${residual}" \
    "${single}" "${distributed}" "${speedup}" "${spmv}" "${dot}" "${axpy}" "${norm}" \
    "${hccl}" "${transfer}" "${sync}" "${allreduce}" "${allgather}" "${error}" \
    "${backend}" "${status}" "${log}" "${threads}" "${omp_min}" "${vector_ops}" >>"${CSV}"
  [[ "${status}" == "pass" ]]
}

# Hardware-free focused check of the parsing/validation logic: synthesizes
# program logs and asserts that (a) a complete log parses to status=pass with
# the parsed single-rank baseline, and (b) logs missing a required field can
# never be recorded as pass. Run with: run_full_scaling.sh --selfcheck
run_selfcheck() {
  local tmp
  tmp="$(mktemp -d)"
  local fake_log="${tmp}/fake.log"
  local fake_csv="${tmp}/fake.csv"
  : >"${fake_csv}"
  cat >"${fake_log}" <<'EOF'
rows = 100
cols = 100
nnz = 500
iteration = 42
RESULT_RESIDUAL=1.2e-08
RESULT_TOTAL_MS=12.345678
Speedup:
1.5x
SpMV = 2.0 ms
Dot = 1.0 ms
AXPY = 0.5 ms
Norm = 0.5 ms
Givens = 0.2 ms
HCCL communication = 3.0 ms
ACL transfer = 1.0 ms
kernel launch = 0.8 ms
synchronization = 0.4 ms
AllReduce calls = 3.000000 (max single-rank per solve)
AllGather calls = 0.000000 (max single-rank per solve)
solution relative error = 4.5e-09
backend = ACL + HCCL
OpenMP threads per rank = 8
OpenMP minimum elements = 4096
vector operations = fused
Single-rank baseline:
total time = 100.000000 ms (single-rank wall time)
Distributed NPU:
total time = 12.345678 ms (MAX across ranks = distributed critical-path wall time; a SUM/world_size rank mean is not a wall time and is never used for total_ms or speedup)
EOF
  local saved_csv="${CSV}"
  local rc=0
  CSV="${fake_csv}"
  if ! write_case "U1" "2" "${fake_log}" "pass" "100.000000"; then
    echo "selfcheck: complete log was not accepted as pass" >&2
    rc=1
  elif [[ "$(tail -n 1 "${fake_csv}" | cut -d, -f8,22)" != "100.000000,pass" ]]; then
    echo "selfcheck: baseline or status column mismatch" >&2
    rc=1
  fi
  : >"${fake_csv}"
  grep -v '^RESULT_RESIDUAL=' "${fake_log}" >"${tmp}/missing_residual.log"
  if write_case "U1" "2" "${tmp}/missing_residual.log" "pass" "100.000000"; then
    echo "selfcheck: log without residual was accepted as pass" >&2
    rc=1
  elif [[ "$(tail -n 1 "${fake_csv}" | cut -d, -f22)" != "fail" ]]; then
    echo "selfcheck: missing-field case not recorded as fail" >&2
    rc=1
  fi
  : >"${fake_csv}"
  grep -v '^backend = ' "${fake_log}" >"${tmp}/missing_backend.log"
  if write_case "U1" "2" "${tmp}/missing_backend.log" "pass" "100.000000"; then
    echo "selfcheck: log without backend was accepted as pass" >&2
    rc=1
  fi
  : >"${fake_csv}"
  sed 's/^RESULT_RESIDUAL=1.2e-08$/RESULT_RESIDUAL=e/' "${fake_log}" >"${tmp}/malformed_number.log"
  if write_case "U1" "2" "${tmp}/malformed_number.log" "pass" "100.000000"; then
    echo "selfcheck: malformed numeric residual ('e') was accepted as pass" >&2
    rc=1
  elif [[ "$(tail -n 1 "${fake_csv}" | cut -d, -f22)" != "fail" ]]; then
    echo "selfcheck: malformed-number case not recorded as fail" >&2
    rc=1
  fi
  : >"${fake_csv}"
  grep -v '^SpMV = ' "${fake_log}" >"${tmp}/missing_stage.log"
  if write_case "U1" "2" "${tmp}/missing_stage.log" "pass" "100.000000"; then
    echo "selfcheck: log without a required stage field (SpMV) was accepted as pass" >&2
    rc=1
  elif [[ "$(tail -n 1 "${fake_csv}" | cut -d, -f22)" != "fail" ]]; then
    echo "selfcheck: missing-stage case not recorded as fail" >&2
    rc=1
  fi
  : >"${fake_csv}"
  sed 's/^backend = ACL + HCCL$/backend = ACL+HCCL/' "${fake_log}" >"${tmp}/wrong_backend.log"
  if write_case "U1" "2" "${tmp}/wrong_backend.log" "pass" "100.000000"; then
    echo "selfcheck: non-exact backend label was accepted as pass" >&2
    rc=1
  fi
  CSV="${saved_csv}"
  rm -rf "${tmp}"
  return "${rc}"
}

if [[ " ${EXTRA[*]:-} " == *" --selfcheck "* ]]; then
  run_selfcheck
  exit $?
fi

if [[ ! -x "${BIN}" ]]; then
  bash "${ROOT_DIR}/scripts/build.sh"
fi
mkdir -p "${ROOT_DIR}/results" "${LOG_DIR}"

# Generate/cache each matrix once before ranks are forked.  This avoids six
# independent ranks racing to create the same CSR1 file on a fresh checkout.
GENERATOR="${ROOT_DIR}/build/bin/matrix_generator"
IFS=',' read -r -a MATRICES <<<"${MATRIX_LIST}"
if [[ ! -x "${GENERATOR}" ]]; then
  echo "matrix_generator is missing: ${GENERATOR}" >&2
  exit 2
fi
for matrix in "${MATRICES[@]}"; do
  if [[ ! -s "${ROOT_DIR}/matrices/${matrix}.csrbin" ]]; then
    "${GENERATOR}" "${ROOT_DIR}/matrices" "${matrix}"
  fi
done

printf '%s\n' \
  'matrix,npus,rows,cols,nnz,iterations,residual,single_ms,distributed_ms,speedup,spmv_ms,dot_ms,axpy_ms,norm_ms,hccl_ms,transfer_ms,sync_ms,allreduce_calls,allgather_calls,solution_error,backend,status,log,threads_per_rank,omp_min_elements,vector_ops' \
  >"${CSV}"

table_for() {
  local npus="$1"
  local variable="RANK_TABLE_${npus}"
  local table="${!variable-}"
  if [[ -n "${TABLE_DIR}" ]]; then
    if [[ -f "${TABLE_DIR}/rank_table_${npus}p.json" ]]; then
      table="${TABLE_DIR}/rank_table_${npus}p.json"
    elif [[ -f "${TABLE_DIR}/rank_table_${npus}.json" ]]; then
      table="${TABLE_DIR}/rank_table_${npus}.json"
    fi
  fi
  # A single RANK_TABLE_FILE is safe only for the matching communicator size.
  if [[ -z "${table}" && "${npus}" == "8" ]]; then
    table="${RANK_TABLE_FILE:-}"
  fi
  printf '%s' "${table}"
}

IFS=',' read -r -a NPUS <<<"${NPU_LIST}"
overall_status=0
for matrix in "${MATRICES[@]}"; do
  baseline_log="${LOG_DIR}/${matrix}_1p_baseline.log"
  echo "===== ${matrix}, single-rank baseline ====="
  set +e
  bash "${ROOT_DIR}/scripts/run.sh" --npus 1 --matrix "${matrix}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" "${EXTRA[@]}" \
    >"${baseline_log}" 2>&1
  baseline_status=$?
  set -e
  baseline_ms="$(awk -F= '/^RESULT_TOTAL_MS=/{value=$2} END{print value}' "${baseline_log}")"
  if [[ "${baseline_status}" -ne 0 || -z "${baseline_ms}" ]]; then
    echo "FAILED: ${matrix}, single-rank baseline; see ${baseline_log}" >&2
    overall_status=1
    continue
  fi
  echo "RESULT_BASELINE_MS=${baseline_ms}"
  for npus in "${NPUS[@]}"; do
    case_log="${LOG_DIR}/${matrix}_${npus}p.log"
    table="$(table_for "${npus}")"
    args=(--npus "${npus}" --no-baseline --single-baseline-ms "${baseline_ms}"
          --matrix "${matrix}" --warmup "${WARMUP}" --repeat "${REPEAT}")
    [[ -n "${table}" ]] && args+=(--rank-table "${table}")
    echo "===== ${matrix}, ${npus} NPU ====="
    set +e
    bash "${ROOT_DIR}/scripts/run.sh" "${args[@]}" "${EXTRA[@]}" >"${case_log}" 2>&1
    case_status=$?
    set -e
    if write_case "${matrix}" "${npus}" "${case_log}" \
        "$([[ ${case_status} -eq 0 ]] && printf pass || printf fail)" "${baseline_ms}"; then
      awk '/^RESULT_TOTAL_MS=|^RESULT_RESIDUAL=/{print}' "${case_log}"
    else
      overall_status=1
      echo "FAILED: ${matrix}, ${npus} NPU; see ${case_log}" >&2
    fi
  done
done

{
  printf '# Dis-GMRES 全矩阵多卡实测结果\n\n'
  printf '> Matrix data and GMRES parameters are identical across runs. Time unit: ms. Only rows with `status=pass` and `backend=ACL + HCCL` are valid Ascend results.\n\n'
  printf '| Matrix | NPU | Threads/rank | Single baseline | Distributed total | HCCL communication | SpMV | Speedup | Error | Status |\n'
  printf '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n'
  awk -F',' 'NR > 1 {printf "| %s | %s | %s | %s | %s | %s | %s | %sx | %s | %s |\n",$1,$2,$24,$8,$9,$15,$11,$10,$20,$22}' "${CSV}"
  printf '\nCSV: `%s`\n' "${CSV}"
  printf 'Logs: `%s`\n' "${LOG_DIR}"
} >"${MARKDOWN}"

echo "saved ${CSV}"
echo "saved ${MARKDOWN}"
exit "${overall_status}"
