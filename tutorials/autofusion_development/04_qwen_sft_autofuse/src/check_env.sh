#!/usr/bin/env bash

set -u

readonly MIN_CANN_BUILD_DATE=20260715
readonly REQUIRED_DISK_KIB=$((30 * 1024 * 1024))
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

failures=0
cann_root="${ASCEND_HOME_PATH:-}"
devices="0,1"
python_input="python3"

pass() {
    printf '[PASS] %s: current=%s; required=%s\n' "$1" "$2" "$3"
}

fail() {
    printf '[FAIL] %s: current=%s; required=%s; action=%s\n' "$1" "$2" "$3" "$4"
    failures=$((failures + 1))
}

usage() {
    printf 'Usage: bash %s [--cann-root <path>] [--devices <id,id>] [--python <executable>]\n' "${BASH_SOURCE[0]}"
}

while (($# > 0)); do
    case "$1" in
        --cann-root)
            if (($# < 2)); then
                fail "arguments" "missing value for --cann-root" "a CANN root path" "provide --cann-root <path>"
                usage
                exit 1
            fi
            cann_root="$2"
            shift 2
            ;;
        --devices)
            if (($# < 2)); then
                fail "arguments" "missing value for --devices" "two device IDs" "provide --devices <id,id>"
                usage
                exit 1
            fi
            devices="$2"
            shift 2
            ;;
        --python)
            if (($# < 2)); then
                fail "arguments" "missing value for --python" "a Python executable" "provide --python <executable>"
                usage
                exit 1
            fi
            python_input="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "arguments" "$1" "known options" "remove the unknown argument"
            usage
            exit 1
            ;;
    esac
done

python_executable=""
python_candidate=""
if [[ "$python_input" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    python_candidate="$(command -v "$python_input" 2>/dev/null || true)"
elif [[ "$python_input" == /* && "$python_input" != *$'\n'* ]]; then
    python_candidate="$python_input"
fi
if [[ -n "$python_candidate" && -f "$python_candidate" && -x "$python_candidate" ]]; then
    python_directory="$(cd -- "$(dirname -- "$python_candidate")" 2>/dev/null && pwd -P)"
    if [[ -n "$python_directory" ]]; then
        canonical_python="$python_directory/$(basename -- "$python_candidate")"
        if [[ "$python_input" != */* || "$python_input" == "$canonical_python" ]]; then
            python_executable="$canonical_python"
        fi
    fi
fi
if [[ -n "$python_executable" ]]; then
    pass "python_interpreter" "$python_executable" "canonical executable file path or single command name"
else
    fail "python_interpreter" "$python_input" "canonical executable file path or single command name" "provide an executable such as --python python3.12"
fi

device_a=""
device_b=""
if [[ "$devices" =~ ^([0-9]+),([0-9]+)$ ]]; then
    raw_device_a="${BASH_REMATCH[1]}"
    raw_device_b="${BASH_REMATCH[2]}"
    while [[ ${#raw_device_a} -gt 1 && "$raw_device_a" == 0* ]]; do
        raw_device_a="${raw_device_a#0}"
    done
    while [[ ${#raw_device_b} -gt 1 && "$raw_device_b" == 0* ]]; do
        raw_device_b="${raw_device_b#0}"
    done
    if [[ ${#raw_device_a} -le 5 && ${#raw_device_b} -le 5 ]]; then
        device_a=$((10#$raw_device_a))
        device_b=$((10#$raw_device_b))
    fi
fi
if [[ -n "$device_a" && -n "$device_b" && "$device_a" != "$device_b" && $device_a -le 65535 && $device_b -le 65535 ]]; then
    pass "devices" "$device_a,$device_b" "two distinct decimal integers in range 0..65535"
else
    device_a=""
    device_b=""
    fail "devices" "$devices" "two distinct decimal integers in range 0..65535" "use a value such as --devices 0,1"
fi

os_name="$(uname -s 2>/dev/null || true)"
architecture="$(uname -m 2>/dev/null || true)"
if [[ "$os_name" == "Linux" && "$architecture" == "aarch64" ]]; then
    pass "system" "$os_name/$architecture" "Linux/aarch64"
else
    fail "system" "${os_name:-unknown}/${architecture:-unknown}" "Linux/aarch64" "use a Linux aarch64 host"
fi

missing_commands=()
required_commands=(bash git npu-smi sha256sum find)
if [[ -n "$python_executable" ]]; then
    required_commands+=("$python_executable")
else
    required_commands+=("$python_input")
fi
for command_name in "${required_commands[@]}"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        missing_commands+=("$command_name")
    fi
done
if ((${#missing_commands[@]} == 0)); then
    pass "commands" "all found" "bash, selected Python, git, npu-smi, sha256sum, find"
else
    fail "commands" "missing: ${missing_commands[*]}" "bash, selected Python, git, npu-smi, sha256sum, find" "install the missing commands and update PATH"
fi

if [[ -n "$python_executable" ]]; then
    python_version="$("$python_executable" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
else
    python_version="unavailable"
fi
if [[ "$python_version" == "3.12" ]]; then
    pass "python" "$python_version" "3.12"
else
    fail "python" "${python_version:-unknown}" "3.12" "activate or install a Python 3.12 environment"
fi

canonical_cann_root=""
if [[ -z "$cann_root" ]]; then
    fail "cann_root" "unset" "a complete CANN root" "provide --cann-root or set ASCEND_HOME_PATH"
elif [[ ! -d "$cann_root" ]]; then
    fail "cann_root" "$cann_root" "an existing directory" "provide the installed CANN root"
elif ! canonical_cann_root="$(cd -- "$cann_root" 2>/dev/null && pwd -P)"; then
    fail "cann_root" "$cann_root" "a resolvable directory" "check path permissions and symlinks"
elif [[ ! -f "$canonical_cann_root/set_env.sh" || ! -d "$canonical_cann_root/opp" || ! -d "$canonical_cann_root/lib64" ]]; then
    fail "cann_root" "$canonical_cann_root" "set_env.sh, opp/, and lib64/" "select a complete CANN installation root"
else
    pass "cann_root" "$canonical_cann_root" "set_env.sh, opp/, and lib64/"
fi

component_names=("Toolkit" "AutoFuse" "Runtime" "HCCL" "OPP")
component_metadata=(
    "share/info/asc-devkit/version.info"
    "share/info/graph_autofusion/version.info"
    "share/info/runtime/version.info"
    "share/info/hccl/version.info"
    "opp/version.info"
)
for index in "${!component_names[@]}"; do
    name="${component_names[$index]}"
    relative="${component_metadata[$index]}"
    metadata_path="${canonical_cann_root:+$canonical_cann_root/}$relative"
    build_date=""
    if [[ -n "$canonical_cann_root" && -f "$metadata_path" ]]; then
        build_date="$("$python_executable" - "$metadata_path" <<'PY'
import re
import sys
from datetime import datetime
from pathlib import Path

field = re.compile(r"(?i)^\s*(?:timestamp|build)\s*=\s*['\"]?([^'\"\s]+)['\"]?\s*$")
value = re.compile(r"(20\d{6})(?:\d{9}|_\d{9})?")
dates = set()
invalid = False
try:
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
except OSError:
    print("missing")
    raise SystemExit
for line in lines:
    match = field.fullmatch(line)
    if not match:
        continue
    timestamp = value.fullmatch(match.group(1))
    if not timestamp:
        invalid = True
        continue
    date = timestamp.group(1)
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        invalid = True
        continue
    dates.add(date)
if invalid:
    print("invalid")
elif not dates:
    print("missing")
elif len(dates) != 1:
    print("conflicting")
else:
    print(dates.pop())
PY
)"
    fi
    if [[ "$build_date" =~ ^[0-9]{8}$ ]] && ((10#$build_date >= MIN_CANN_BUILD_DATE)); then
        pass "CANN $name" "$build_date" ">=$MIN_CANN_BUILD_DATE"
    else
        fail "CANN $name" "${build_date:-missing}" ">=$MIN_CANN_BUILD_DATE" "install a current $name component with valid version.info"
    fi
done

disk_output="$(df -Pk -- "$SCRIPT_DIR" 2>/dev/null || true)"
disk_line="${disk_output##*$'\n'}"
read -r _filesystem _blocks _used disk_available _capacity _mount <<< "$disk_line"
if [[ "${disk_available:-}" =~ ^[0-9]+$ ]] && ((disk_available >= REQUIRED_DISK_KIB)); then
    pass "disk" "${disk_available} KiB free" ">=$REQUIRED_DISK_KIB KiB free"
else
    fail "disk" "${disk_available:-unknown} KiB free" ">=$REQUIRED_DISK_KIB KiB free" "free at least 30 GiB on the example filesystem"
fi

npu_info=""
npu_mapping_info=""
if command -v npu-smi >/dev/null 2>&1; then
    npu_info="$(npu-smi info 2>&1)"
    npu_status=$?
    npu_mapping_info="$(npu-smi info -m 2>&1)"
    npu_mapping_status=$?
else
    npu_status=127
    npu_mapping_status=127
fi
parsed_mapping=""
mapping_parse_status=1
if [[ $npu_mapping_status -eq 0 && -n "$python_executable" ]]; then
    if parsed_mapping="$("$python_executable" - "$npu_mapping_info" <<'PY'
import re
import sys

header = re.compile(r"^\s*NPU ID\s+Chip ID\s+Chip Logic ID\s+Chip Phy-ID\s+Chip Name\s*$")
row = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+|-)\s+(\d+|-)\s+(\S(?:.*\S)?)\s*$")
lines = sys.argv[1].splitlines()
header_index = next((index for index, line in enumerate(lines) if header.fullmatch(line)), None)
if header_index is None:
    raise SystemExit(1)
mapping = {}
for line in lines[header_index + 1 :]:
    if not line.strip():
        continue
    match = row.fullmatch(line)
    if not match:
        raise SystemExit(1)
    if match.group(3) == "-" and match.group(4) == "-":
        continue
    if "-" in (match.group(3), match.group(4)):
        raise SystemExit(1)
    logic_id = int(match.group(3), 10)
    physical_id = int(match.group(4), 10)
    if logic_id > 65535 or physical_id > 65535 or logic_id in mapping:
        raise SystemExit(1)
    mapping[logic_id] = physical_id
if not mapping:
    raise SystemExit(1)
for logic_id in sorted(mapping):
    print(f"{logic_id} {mapping[logic_id]}")
PY
)"; then
        mapping_parse_status=0
    fi
fi

logic_ids=()
physical_ids=()
mapping_summary=()
while read -r logic_id physical_id; do
    if [[ "$logic_id" =~ ^[0-9]+$ && "$physical_id" =~ ^[0-9]+$ ]]; then
        logic_ids+=("$logic_id")
        physical_ids+=("$physical_id")
        mapping_summary+=("$logic_id->$physical_id")
    fi
done <<< "$parsed_mapping"
mapped_nodes=()
selected_physical_ids=()
if [[ -n "$device_a" ]]; then
    for device_id in "$device_a" "$device_b"; do
        for index in "${!logic_ids[@]}"; do
            if ((device_id == 10#${logic_ids[$index]})); then
                physical_id="${physical_ids[$index]}"
                expected_node="/dev/davinci$physical_id"
                actual_node="$(find /dev -maxdepth 1 -type c -name "davinci$physical_id" -print 2>/dev/null)"
                if [[ "$actual_node" == "$expected_node" ]]; then
                    mapped_nodes+=("logical $device_id=$expected_node")
                    selected_physical_ids+=("$physical_id")
                fi
                break
            fi
        done
    done
fi
physical_ids_distinct=0
if [[ ${#selected_physical_ids[@]} -eq 2 && "${selected_physical_ids[0]}" != "${selected_physical_ids[1]}" ]]; then
    physical_ids_distinct=1
fi
if [[ -n "$device_a" && $npu_status -eq 0 && $npu_mapping_status -eq 0 && $mapping_parse_status -eq 0 && ${#mapped_nodes[@]} -eq 2 && $physical_ids_distinct -eq 1 ]]; then
    npu_summary="${npu_info%%$'\n'*}"
    pass "npu" "${npu_summary:-npu-smi succeeded}; logical-to-physical=${mapping_summary[*]}; ${mapped_nodes[*]}" "npu-smi info and info -m succeed, requested logical IDs map to /dev character devices"
else
    if [[ ${#selected_physical_ids[@]} -eq 2 && $physical_ids_distinct -eq 0 ]]; then
        npu_required="selected logical IDs map to distinct physical devices"
    else
        npu_required="npu-smi info and info -m succeed, both logical IDs exist and mapped /dev nodes are character devices"
    fi
    fail "npu" "info status=$npu_status; info -m status=$npu_mapping_status; mapping parse status=$mapping_parse_status; logical-to-physical=${mapping_summary[*]:-none}; mapped nodes=${mapped_nodes[*]:-none}; requested=$devices" "$npu_required" "check npu-smi mapping, driver device nodes, and permissions"
fi

if ((failures > 0)); then
    exit 1
fi
exit 0
