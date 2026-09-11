#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_case.sh --mode static|sk [--device-id ID] [--warmup N] [--output DIR]
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
network_script="$script_dir/network_fragment_4ops.py"
static_kernel_output_dir="$script_dir/static_kernel_compile_outputs"
run_log="$(mktemp "$script_dir/.run_case.XXXXXX.log")"
mode=""
device_id=0
warmup=1
output_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        --device-id) device_id="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --output) output_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$mode" in
    static) run_mode="aclgraph" ;;
    sk) run_mode="aclgraph+sk" ;;
    *) echo "--mode must be static or sk" >&2; exit 2 ;;
esac

if [[ -z "$output_dir" ]]; then
    output_dir="$script_dir/profiling-$mode"
fi

if ! command -v msprof >/dev/null 2>&1; then
    echo "Cannot find msprof. Source the CANN environment first." >&2
    exit 1
fi

cleanup_static_kernel_packages() {
    local uninstall_script
    while IFS= read -r uninstall_script; do
        if [[ -n "$uninstall_script" && -f "$uninstall_script" ]]; then
            bash "$uninstall_script" >/dev/null 2>&1 || \
                echo "Warning: failed to uninstall static kernel package: $uninstall_script" >&2
        fi
    done < <(grep -Eo '/[^[:space:]]+/uninstall\.sh' "$run_log" | sort -u || true)
}

cleanup() {
    bash "$script_dir/ops/aclgraph_clear_ops/uninstall.sh" >/dev/null 2>&1 || true
    cleanup_static_kernel_packages
    rm -f "$run_log"
}
trap cleanup EXIT

cd "$script_dir"
rm -rf "$static_kernel_output_dir"
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"
bash "$script_dir/ops/aclgraph_clear_ops/install.sh"

echo "DEVICE_ID=$device_id"
echo "RUN_MODE=$run_mode"
echo "NETWORK_SCRIPT=$network_script"
echo "NETWORK_CASE=6 operators x 50 layers"
echo "MSPROF_OUTPUT=$output_dir"

python_cmd="${PYTHON:-python3}"
set +e
msprof \
    --output="$output_dir" \
    --ascendcl=on \
    --runtime-api=on \
    --task-time=l1 \
    --ai-core=on \
    --aic-mode=task-based \
    --aic-metrics=PipeUtilization \
    --msproftx=on \
    "$python_cmd" "$network_script" \
        --run_mode "$run_mode" \
        --device-id "$device_id" \
        --warmup "$warmup" \
        --step_cnt 1 \
        --msprof 2>&1 | tee "$run_log"
msprof_rc=${PIPESTATUS[0]}
set -e
if [[ "$msprof_rc" -ne 0 ]]; then
    exit "$msprof_rc"
fi

summary_args=("$output_dir" "--case-name" "$run_mode")
if [[ "$mode" == "sk" ]]; then
    summary_args+=("--baseline-root" "$script_dir/profiling-static")
fi
"$python_cmd" "$script_dir/summarize_profile.py" "${summary_args[@]}"
