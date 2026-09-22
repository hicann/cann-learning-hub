#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_q1.sh [--device-id ID] [--warmup N] [--steps N] [--output DIR] [--profiler | --no-profiler]

Run the question 1 static-kernel ACLGraph sample.
Profiling is disabled by default. Use --profiler to collect with msprof.

Defaults:
  --device-id ${ASCEND_DEVICE_ID:-${NPU_DEVICE_ID:-0}}
  --warmup    1
  --steps     1
  --output    ./profiling
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
device_id="${ASCEND_DEVICE_ID:-${NPU_DEVICE_ID:-0}}"
profiler_enabled=false
warmup=1
steps=1
output_dir="${script_dir}/profiling"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-id)
            [[ $# -ge 2 ]] || { echo "--device-id requires a value" >&2; exit 2; }
            device_id="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "--warmup requires a value" >&2; exit 2; }
            warmup="$2"
            shift 2
            ;;
        --steps)
            [[ $# -ge 2 ]] || { echo "--steps requires a value" >&2; exit 2; }
            steps="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { echo "--output requires a value" >&2; exit 2; }
            output_dir="$2"
            shift 2
            ;;
        --profiler)
            profiler_enabled=true
            shift
            ;;
        --no-profiler)
            profiler_enabled=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$script_dir"

cann_home=""
for candidate in \
    "${ASCEND_HOME_PATH:-}" \
    "${ASCEND_TOOLKIT_HOME:-}" \
    "/usr/local/Ascend/cann"; do
    if [[ -n "$candidate" && -f "$candidate/set_env.sh" ]]; then
        cann_home="$(realpath "$candidate")"
        break
    fi
done
if [[ -z "$cann_home" ]]; then
    echo "Cannot locate CANN set_env.sh; set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME first." >&2
    exit 1
fi

restore_nounset=false
if [[ "$-" == *u* ]]; then
    restore_nounset=true
    set +u
fi
# shellcheck disable=SC1090
source "$cann_home/set_env.sh"
if [[ "$restore_nounset" == true ]]; then
    set -u
fi

if [[ "$profiler_enabled" == true ]]; then
    if ! command -v msprof >/dev/null 2>&1; then
        echo "Cannot find msprof after sourcing the CANN environment." >&2
        exit 1
    fi
    mkdir -p "$output_dir"
    output_dir="$(realpath "$output_dir")"
fi

export ASCEND_DEVICE_ID="$device_id"
export NPU_DEVICE_ID="$device_id"
export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1

echo "CANN_HOME=$cann_home"
echo "DEVICE_ID=$device_id"

model_command=(python3 "$script_dir/sample_network_segment.py"
    --device-id "$device_id" --warmup "$warmup" --steps "$steps")

if [[ "$profiler_enabled" == true ]]; then
    echo "MSPROF_OUTPUT=$output_dir"
    run_command=(msprof
        --output="$output_dir"
        --ascendcl=on
        --runtime-api=on
        --task-time=l1
        --ai-core=on
        --aic-mode=task-based
        --aic-metrics=PipeUtilization
        "${model_command[@]}")
else
    echo "MSPROF=disabled"
    run_command=("${model_command[@]}")
fi

# shellcheck disable=SC1091
source "$script_dir/ops/common.sh"

PYTHONDONTWRITEBYTECODE=1 run_with_clear_ops "${run_command[@]}"
