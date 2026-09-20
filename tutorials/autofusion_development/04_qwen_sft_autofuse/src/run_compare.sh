#!/usr/bin/env bash
set -u

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
    printf 'Usage: %s --cann-root PATH --model PATH --data PATH [--devices IDS] [--output-dir PATH]\n' "$0" >&2
}

cann_root=""
model=""
data=""
devices="0,1"
output_dir="results"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cann-root|--model|--data|--devices|--output-dir)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            option=$1
            value=$2
            case "$option" in
                --cann-root) cann_root=$value ;;
                --model) model=$value ;;
                --data) data=$value ;;
                --devices) devices=$value ;;
                --output-dir) output_dir=$value ;;
            esac
            shift 2
            ;;
        *) usage; exit 2 ;;
    esac
done

if [ -z "$cann_root" ] || [ -z "$model" ] || [ -z "$data" ]; then
    usage
    exit 2
fi
if [[ ! "$devices" =~ ^([0-9]+),([0-9]+)$ ]]; then
    printf 'devices must contain exactly two non-negative integer IDs: %s\n' "$devices" >&2
    exit 2
fi
if [ "${#BASH_REMATCH[1]}" -gt 5 ] || [ "${#BASH_REMATCH[2]}" -gt 5 ]; then
    printf 'devices must contain two distinct IDs in range 0..65535: %s\n' "$devices" >&2
    exit 2
fi
device0=$((10#${BASH_REMATCH[1]}))
device1=$((10#${BASH_REMATCH[2]}))
if [ "$device0" -gt 65535 ] || [ "$device1" -gt 65535 ] || [ "$device0" -eq "$device1" ]; then
    printf 'devices must contain two distinct IDs in range 0..65535: %s\n' "$devices" >&2
    exit 2
fi
devices="$device0,$device1"
if [ ! -d "$cann_root" ] || [ ! -f "$cann_root/set_env.sh" ]; then
    printf 'CANN root must contain set_env.sh: %s\n' "$cann_root" >&2
    exit 2
fi
if [ ! -d "$model" ] || [ ! -f "$data" ]; then
    printf 'model or data path does not exist\n' >&2
    exit 2
fi
cann_root=$(readlink -f -- "$cann_root") || exit 2
model=$(readlink -f -- "$model") || exit 2
data=$(readlink -f -- "$data") || exit 2

python_path=$(command -v python3 2>/dev/null || true)
if [ -z "$python_path" ] || [ ! -f "$python_path" ] || [ ! -x "$python_path" ]; then
    printf 'python3 must resolve to an executable file\n' >&2
    exit 2
fi
python_dir=$(cd -- "$(dirname -- "$python_path")" 2>/dev/null && pwd -P) || exit 2
python_path="$python_dir/$(basename -- "$python_path")"

if ! bash "$SCRIPT_DIR/check_env.sh" --cann-root "$cann_root" --devices "$devices" --python "$python_path"; then
    printf 'preflight environment check failed\n' >&2
    exit 2
fi
if ! (
    unset LD_PRELOAD PYTHONHOME CONDA_PREFIX
    # shellcheck disable=SC1090
    source "$cann_root/set_env.sh" || exit 1
    export ASCEND_RT_VISIBLE_DEVICES="$devices"
    "$python_path" -c '
import torch
import torch_npu
import torchtitan
import torchtitan_npu
import triton
import inductor_npu_ext

count = torch.npu.device_count()
if count != 2:
    raise RuntimeError(f"expected exactly 2 visible NPU devices, got {count}")
'
); then
    printf 'preflight Python import or two-device smoke check failed\n' >&2
    exit 2
fi

if [ -L "$output_dir" ]; then
    printf 'output root must not be a symlink: %s\n' "$output_dir" >&2
    exit 2
fi
if [ -e "$output_dir" ] && [ ! -d "$output_dir" ]; then
    printf 'output root is not a directory: %s\n' "$output_dir" >&2
    exit 2
fi
mkdir -p -- "$output_dir" || exit 2
output_dir=$(readlink -f -- "$output_dir") || exit 2
if [ "$(stat -c %u -- "$output_dir")" != "$(id -u)" ]; then
    printf 'output root must be owned by current user: %s\n' "$output_dir" >&2
    exit 2
fi
output_mode=$(stat -c %a -- "$output_dir") || exit 2
if (( (8#$output_mode & 8#022) != 0 )); then
    printf 'output root must not be group or world writable: %s\n' "$output_dir" >&2
    exit 2
fi

umask 077
result_prefix="$output_dir/$(date -u +%Y%m%dT%H%M%SZ)-$$"
result_dir=""
for attempt in {0..15}; do
    candidate="$result_prefix-$attempt"
    if mkdir -m 0700 -- "$candidate" 2>/dev/null; then
        result_dir=$candidate
        break
    fi
done
if [ -z "$result_dir" ]; then
    printf 'failed to create result directory after 16 attempts: %s\n' "$output_dir" >&2
    exit 2
fi
result_identity=$(stat -Lc '%d:%i' -- "$result_dir") || exit 2
check_result_dir() {
    [ -d "$result_dir" ] && [ ! -L "$result_dir" ] &&
        [ "$(stat -Lc '%d:%i' -- "$result_dir" 2>/dev/null)" = "$result_identity" ] || {
        printf 'result directory changed during execution\n' >&2
        return 1
    }
}
check_result_dir || exit 2
mkdir -m 0700 -- "$result_dir/profiling_off" "$result_dir/profiling_on" \
    "$result_dir/torch_extensions" "$result_dir/triton_cache" "$result_dir/cache" || exit 2
for cache_kind in off_train on_train off_profile on_profile; do
    mkdir -m 0700 -- "$result_dir/cache/$cache_kind" || exit 2
done

train_command="${AF_SFT_TRAIN_COMMAND:-}"
if [ -n "$train_command" ]; then
    case "$train_command" in
        /*) ;;
        *) printf 'AF_SFT_TRAIN_COMMAND must be an absolute executable path without arguments\n' >&2; exit 2 ;;
    esac
    train_command=$(readlink -f -- "$train_command") || exit 2
    [ -f "$train_command" ] && [ -x "$train_command" ] || {
        printf 'AF_SFT_TRAIN_COMMAND is not executable: %s\n' "$train_command" >&2
        exit 2
    }
else
    [ -n "${TORCHTITAN_NPU_DIR:-}" ] || {
        printf 'TORCHTITAN_NPU_DIR is required when AF_SFT_TRAIN_COMMAND is unset\n' >&2
        exit 2
    }
    torchtitan_dir=$(readlink -f -- "$TORCHTITAN_NPU_DIR" 2>/dev/null) || {
        printf 'invalid TORCHTITAN_NPU_DIR: %s\n' "$TORCHTITAN_NPU_DIR" >&2
        exit 2
    }
    runner_path="$torchtitan_dir/scripts/run_train.sh"
    train_command=$(readlink -f -- "$runner_path" 2>/dev/null) || {
        printf 'missing regular file %s/scripts/run_train.sh\n' "$TORCHTITAN_NPU_DIR" >&2
        exit 2
    }
    [ "$train_command" = "$runner_path" ] && [ -f "$train_command" ] || {
        printf 'default runner must be a canonical regular file inside TORCHTITAN_NPU_DIR/scripts: %s\n' \
            "$runner_path" >&2
        exit 2
    }
fi

python_base_lib="${PYTHON_BASE_LIB:-}"
python_venv_config=""
if [ -n "$python_path" ]; then
    python_venv_config=$(readlink -f -- "$(dirname -- "$python_path")/../pyvenv.cfg" 2>/dev/null || true)
fi
if [ -z "$python_base_lib" ] && [ -n "$python_venv_config" ] && [ -f "$python_venv_config" ]; then
    python_home=$(while IFS='=' read -r key value; do
        if [ "${key// /}" = home ]; then printf '%s' "${value# }"; break; fi
    done < "$python_venv_config")
    if [ -n "$python_home" ]; then
        python_base_lib=$(readlink -f -- "$python_home/../lib" 2>/dev/null || true)
    fi
fi

run_training() {
    local mode=$1 profile=$2 steps=$3 log=$4 profile_dir=$5
    local run_kind="${mode}_${profile}"
    local inductor_cache
    local seed
    check_result_dir || return 125
    inductor_cache=$(readlink -f -- "$result_dir/cache/$run_kind") || return 125
    (
        # shellcheck disable=SC1090
        source "$cann_root/set_env.sh" || exit 126
        export ASCEND_RT_VISIBLE_DEVICES="$devices"
        export NGPU=2
        export MODULE=torchtitan_npu.models.qwen3
        export CONFIG=sft_qwen3_1_7b_wordle
        export TORCH_EXTENSIONS_DIR="$result_dir/torch_extensions"
        export TRITON_CACHE_DIR="$result_dir/triton_cache"
        export TORCHINDUCTOR_CACHE_DIR="$inductor_cache"
        export TORCHINDUCTOR_SIZE_ASSERTS=0
        export TORCHINDUCTOR_FX_GRAPH_CACHE=0
        export TORCHINDUCTOR_AUTOGRAD_CACHE=0
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HF_HUB_DISABLE_XET=1
        export TRAIN_SEQUENCE_LENGTH=1024
        export LOG_RANK=0
        if [ -n "$python_base_lib" ]; then
            export PYTHON_BASE_LIB="$python_base_lib"
            export LD_LIBRARY_PATH="$python_base_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        if [ -n "${AF_SFT_TRAIN_COMMAND:-}" ]; then
            "$train_command" "$mode" "$profile" "$steps" "$log" "$profile_dir" "$model" "$data" "$devices"
        else
            seed=42
            if [ "$profile" = profile ]; then seed=43; fi
            args=(--hf_assets_path "$model" --checkpoint.load_only --training.global_batch_size 4
                --training.steps "$steps" --debug.seed "$seed")
            if [ "$profile" = profile ]; then
                args+=(--profiling.enable_profiling --profiling.profile_step_start 6
                    --profiling.profile_step_end 11 --profiling.save_traces_folder "$profile_dir")
            fi
            if [ "$mode" = on ]; then args+=(--compile.enable); fi
            args+=(dataloader:chat-data-loader-config --dataloader.dataset-path parquet
                --dataloader.data-files "$data")
            cd "$torchtitan_dir" || exit 127
            bash scripts/run_train.sh "${args[@]}"
        fi
    ) > "$log" 2>&1
}

off_train_status=0
on_train_status=0
off_profile_status=0
on_profile_status=0
run_training off train 10 "$result_dir/af_off.log" "$result_dir/profiling_off" || off_train_status=$?
if [ "$off_train_status" -eq 0 ]; then
    run_training on train 10 "$result_dir/af_on.log" "$result_dir/profiling_on" || on_train_status=$?
else
    check_result_dir || exit 2
    : > "$result_dir/af_on.log"
    on_train_status=125
fi
if [ "$off_train_status" -eq 0 ] && [ "$on_train_status" -eq 0 ]; then
    run_training off profile 12 "$result_dir/profiling_off.log" "$result_dir/profiling_off" || off_profile_status=$?
    run_training on profile 12 "$result_dir/profiling_on.log" "$result_dir/profiling_on" || on_profile_status=$?
else
    check_result_dir || exit 2
    : > "$result_dir/profiling_off.log"
    check_result_dir || exit 2
    : > "$result_dir/profiling_on.log"
    off_profile_status=125
    on_profile_status=125
fi

check_result_dir || exit 2
report_tmp="$result_dir/.report.md.tmp"
set -C
"$python_path" - "$result_dir" "$off_train_status" "$on_train_status" "$off_profile_status" "$on_profile_status" \
    "$model" "$data" "$cann_root" "$devices" \
    > "$report_tmp" <<'PY'
import csv
import hashlib
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
off_train_status, on_train_status, off_profile_status, on_profile_status = map(int, sys.argv[2:6])
model_path, data_path, cann_path, devices = sys.argv[6:10]
ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
step_re = re.compile(
    r"(?<![\w.])step\s*:\s*(\d+)(?![\w.]).*?"
    r"(?<![\w.])elapsed_time_per_step\s*:\s*([^\s,]+)",
    re.IGNORECASE,
)
expected_steps = set(range(6, 11))


def parse_steps(path):
    values = {}
    duplicate = False
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        match = step_re.search(ansi_re.sub("", raw_line))
        if not match:
            continue
        step = int(match.group(1))
        if step not in expected_steps:
            continue
        if step in values:
            duplicate = True
        token = match.group(2)
        if token.lower().endswith("s"):
            token = token[:-1]
        try:
            value = float(token)
        except ValueError:
            return None
        values[step] = value
    if duplicate or set(values) != expected_steps:
        return None
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        return None
    return values


off_steps = parse_steps(root / "af_off.log")
on_steps = parse_steps(root / "af_on.log")
training_valid = (
    off_train_status == 0 and on_train_status == 0 and off_steps is not None and on_steps is not None
)
if training_valid:
    off_average = sum(off_steps.values()) / len(expected_steps)
    on_average = sum(on_steps.values()) / len(expected_steps)
    gain = (off_average - on_average) / off_average * 100
    conclusion = "AF_ON_FASTER" if gain > 0 else "NO_GAIN"
else:
    off_average = on_average = gain = None
    conclusion = "RUN_FAILED"

phases = ["Computing", "Communication(Not Overlapped)", "Overlapped", "Free", "Stage"]
required_columns = {"Device_id", "Step", *phases}
profile_steps = set(range(5, 10))


def parse_profile(directory, exit_status):
    if exit_status != 0:
        return None, "UNAVAILABLE", None
    files = list(directory.rglob("step_trace_time.csv"))
    if len(files) != 1:
        return None, "UNAVAILABLE", None
    try:
        with files[0].open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not required_columns.issubset(reader.fieldnames or []):
                return None, "INCOMPARABLE", None
            rows = list(reader)
        if len(rows) != 5:
            return None, "INCOMPARABLE", None
        steps = [int(row["Step"]) for row in rows]
        if set(steps) != profile_steps or len(set(steps)) != len(steps):
            return None, "INCOMPARABLE", None
        device_ids = {row["Device_id"] for row in rows}
        if len(device_ids) != 1:
            return None, "INCOMPARABLE", None
        samples = {phase: [float(row[phase]) for row in rows] for phase in phases}
        if any(
            not math.isfinite(value) or value < 0
            for values in samples.values()
            for value in values
        ):
            return None, "INCOMPARABLE", None
        averages = {phase: sum(values) / len(values) for phase, values in samples.items()}
        return averages, "VALID", device_ids.pop()
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return None, "INCOMPARABLE", None


off_profile, off_profile_state, off_device_id = parse_profile(root / "profiling_off", off_profile_status)
on_profile, on_profile_state, on_device_id = parse_profile(root / "profiling_on", on_profile_status)
if "UNAVAILABLE" in (off_profile_state, on_profile_state):
    profile_status = "UNAVAILABLE"
elif off_profile_state != "VALID" or on_profile_state != "VALID" or off_device_id != on_device_id:
    profile_status = "INCOMPARABLE"
else:
    profile_status = "COMPARABLE"

computing_gain = None
stage_total_gain = None
stage_minus_computing_gain = None
off_stage_minus_computing = None
on_stage_minus_computing = None

if profile_status == "COMPARABLE":
    off_computing = off_profile["Computing"]
    on_computing = on_profile["Computing"]
    off_stage = off_profile["Stage"]
    on_stage = on_profile["Stage"]

    if off_computing > 0:
        computing_gain = (off_computing - on_computing) / off_computing * 100

    if off_stage > 0:
        stage_total_gain = (off_stage - on_stage) / off_stage * 100

    off_stage_minus_computing = off_stage - off_computing
    on_stage_minus_computing = on_stage - on_computing

    if off_stage_minus_computing > 0:
        stage_minus_computing_gain = (
            (off_stage_minus_computing - on_stage_minus_computing)
            / off_stage_minus_computing
            * 100
        )

print("# AF SFT 对比报告\n")
print("## 环境摘要\n")
for label, path in (("model", model_path), ("data", data_path), ("CANN", cann_path)):
    digest = hashlib.sha256(path.encode()).hexdigest()[:16]
    print(f"- {label} ({Path(path).name}): `{digest}`")
metadata = {
    "Toolkit": "share/info/asc-devkit/version.info",
    "AutoFuse": "share/info/graph_autofusion/version.info",
    "Runtime": "share/info/runtime/version.info",
    "HCCL": "share/info/hccl/version.info",
    "OPP": "opp/version.info",
}
date_re = re.compile(r"(?im)^\s*(?:timestamp|build)\s*=\s*['\"]?(20\d{6})")
for component, relative in metadata.items():
    dates = set(date_re.findall((Path(cann_path) / relative).read_text(errors="replace")))
    print(f"- CANN {component}: `{dates.pop() if len(dates) == 1 else 'unknown'}`")
print(f"- devices: `{devices}`")
print("\n## 运行参数\n")
print("- global batch size: `4`")
print("- sequence length: `1024`")
print("- training steps: `10`")
print("- profiling steps: `12`")
print("- training seed: `42`")
print("- profiling seed: `43`")
print("\n## 退出状态\n")
print(f"- AF OFF training: `{off_train_status}`")
print(f"- AF ON training: `{on_train_status}`")
print(f"- AF OFF profiling: `{off_profile_status}`")
print(f"- AF ON profiling: `{on_profile_status}`")
print("\n## 结果\n")
print(f"- 训练结论：**{conclusion}**")
print(f"- Profiling 状态：**{profile_status}**")
if training_valid:
    print(f"- OFF 平均耗时：{off_average:.6g} s")
    print(f"- ON 平均耗时：{on_average:.6g} s")
    print(f"- 训练收益：{gain:.6g}%")
else:
    print("- 训练结论因运行失败、步骤重复/缺失或耗时不是有限正数而不可用。")
print("\n## 指标说明\n")
print("训练耗时是 active steps 6-10 的完整 step 墙钟均值，单位 s，越低越好。")
print("训练收益按 `(OFF - ON) / OFF * 100%` 计算，正值表示 AF ON 更快。")
print("Profiler 参数 start=6、end=11 对应 active steps 6-10；CSV Step 5-9 逐行映射这些训练步，并按列取均值，阶段单位固定 us。")
print("Profiling 仅在两侧命令成功且 CSV 字段、聚合 Step 和数值均有效时为 COMPARABLE；它不覆盖训练结论。")
print("\n## 阶段聚合\n")
for phase in phases:
    if profile_status == "COMPARABLE":
        print(f"- {phase}: OFF {off_profile[phase]:.6g} us, ON {on_profile[phase]:.6g} us")
    else:
        print(f"- {phase}: 不可比较")

print("\n## Profiling 收益计算\n")
if profile_status == "COMPARABLE":
    print("### Computing 收益\n")
    print(
        f"- 计算过程：`({off_profile['Computing']:.6g} - {on_profile['Computing']:.6g}) "
        f"/ {off_profile['Computing']:.6g} × 100%`"
    )
    if computing_gain is not None:
        print(f"- 计算结果：**{computing_gain:.6g}%**")
    else:
        print("- 计算结果：不可用")

    print("\n### Stage Total 收益\n")
    print(
        f"- 计算过程：`({off_profile['Stage']:.6g} - {on_profile['Stage']:.6g}) "
        f"/ {off_profile['Stage']:.6g} × 100%`"
    )
    if stage_total_gain is not None:
        print(f"- 计算结果：**{stage_total_gain:.6g}%**")
    else:
        print("- 计算结果：不可用")

    print("\n### Stage-Computing 收益\n")
    print(
        f"- OFF Stage-Computing：`{off_profile['Stage']:.6g} - "
        f"{off_profile['Computing']:.6g} = {off_stage_minus_computing:.6g} us`"
    )
    print(
        f"- ON Stage-Computing：`{on_profile['Stage']:.6g} - "
        f"{on_profile['Computing']:.6g} = {on_stage_minus_computing:.6g} us`"
    )
    if stage_minus_computing_gain is not None:
        print(
            f"- 计算过程：`({off_stage_minus_computing:.6g} - "
            f"{on_stage_minus_computing:.6g}) / "
            f"{off_stage_minus_computing:.6g} × 100%`"
        )
        print(f"- 计算结果：**{stage_minus_computing_gain:.6g}%**")
    else:
        print("- 计算结果：不可用")
else:
    print("Profiling 数据不可比较，无法计算阶段收益。")

print("\n这是单轮 Quick 方向性证据，不等价于多轮正式收益；阶段时间不能直接归因于整网收益。")
sys.exit(0 if training_valid else 1)
PY
report_status=$?
set +C
check_result_dir || exit 2
mv -- "$report_tmp" "$result_dir/report.md" || exit 2
if [ "$report_status" -ne 0 ]; then exit 1; fi
exit 0
