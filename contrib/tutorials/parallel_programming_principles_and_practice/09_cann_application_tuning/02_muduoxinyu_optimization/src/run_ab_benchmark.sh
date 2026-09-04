#!/usr/bin/env bash
# Run the fixed MuduoXinyu implementation-package A/B experiment.
set -euo pipefail

usage() {
  echo "Usage: bash run_ab_benchmark.sh --muduo-root <dir> --model <file> --tokenizer <file> --output-dir <new-dir> --build-log <file>" >&2
  echo "  --build-log is mandatory: the executed binary must be bound to THIS build's log." >&2
  exit 2
}

MUDUO_ROOT=""
MODEL=""
TOKENIZER=""
OUTPUT=""
BUILD_LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --muduo-root) MUDUO_ROOT="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --tokenizer) TOKENIZER="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT="${2:-}"; shift 2 ;;
    --build-log) BUILD_LOG="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$MUDUO_ROOT" && -n "$MODEL" && -n "$TOKENIZER" && -n "$OUTPUT" && -n "$BUILD_LOG" ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sha256_file() {
  sha256sum -- "$1" | sed 's/^\\//' | awk '{print $1}'
}

file_size() {
  wc -c < "$1" | tr -d ' '
}

fail_gate() {
  echo "AB_EVIDENCE_GATE=FAIL $*" >&2
  exit 2
}

# All executed objects are bound to the manifest by canonical absolute path.
MUDUO_ROOT="$(readlink -f -- "$MUDUO_ROOT")"
MODEL="$(readlink -f -- "$MODEL")"
TOKENIZER="$(readlink -f -- "$TOKENIZER")"

BINARY="$MUDUO_ROOT/muduoXinyu"
git -C "$MUDUO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Not a git repository: $MUDUO_ROOT" >&2
  exit 2
}
for file in "$BINARY" "$MODEL" "$TOKENIZER"; do
  [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
[[ -x "$BINARY" ]] || { echo "Binary is not executable: $BINARY" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Output directory already exists: $OUTPUT" >&2; exit 2; }
# --build-log is mandatory: an old binary must not pass on fresh source. The
# file must exist, be non-empty, and be bound by canonical absolute path.
BUILD_LOG="$(readlink -f -- "$BUILD_LOG")"
if [[ ! -f "$BUILD_LOG" || ! -s "$BUILD_LOG" ]]; then
  echo "Build log missing or empty: $BUILD_LOG" >&2
  exit 2
fi

# ---------- build-source evidence gate ----------
# The executed binary must be bound to the course candidate: HEAD must equal the
# apply_patch.sh baseline, the patch file must match the verified hash, and the
# key patched source (npuBackend.cpp) must match the verified candidate hash.
# Constants are read from apply_patch.sh so this gate can never drift from the
# script that actually applies the patch. Any failure exits nonzero here, so
# RUN_AB=PASS can never be printed without the declared build-source evidence.
BASELINE_COMMIT="$(awk -F'"' '/^BASELINE_COMMIT=/{print $2; exit}' "$SCRIPT_DIR/apply_patch.sh")"
PATCH_SHA256_EXPECTED="$(awk -F'"' '/^PATCH_SHA256=/{print $2; exit}' "$SCRIPT_DIR/apply_patch.sh")"
CANDIDATE_NPUBACKEND_SHA256="$(awk -F'"' '/^CANDIDATE_NPUBACKEND_SHA256=/{print $2; exit}' "$SCRIPT_DIR/apply_patch.sh")"
if [[ -z "$BASELINE_COMMIT" || -z "$PATCH_SHA256_EXPECTED" || -z "$CANDIDATE_NPUBACKEND_SHA256" ]]; then
  fail_gate "cannot read the baseline/patch/candidate constants from apply_patch.sh"
fi
COMMIT="$(git -C "$MUDUO_ROOT" rev-parse HEAD)"
if [[ "$COMMIT" != "$BASELINE_COMMIT" ]]; then
  fail_gate "HEAD($COMMIT) != course baseline($BASELINE_COMMIT)"
fi
PATCH_FILE="$SCRIPT_DIR/patch/muduoxinyu_flashattention_v1.patch"
if [[ ! -f "$PATCH_FILE" ]]; then
  fail_gate "patch file missing: $PATCH_FILE"
fi
PATCH_SHA256="$(sha256_file "$PATCH_FILE")"
if [[ "$PATCH_SHA256" != "$PATCH_SHA256_EXPECTED" ]]; then
  fail_gate "patch hash mismatch: actual=$PATCH_SHA256 expected=$PATCH_SHA256_EXPECTED"
fi
NPU_CPP="$MUDUO_ROOT/src/backend/npuBackend.cpp"
NPU_HPP="$MUDUO_ROOT/src/backend/npuBackend.hpp"
for key_source in "$NPU_CPP" "$NPU_HPP"; do
  [[ -f "$key_source" ]] || fail_gate "key source missing: $key_source"
done
NPU_CPP_SHA256="$(sha256_file "$NPU_CPP")"
if [[ "$NPU_CPP_SHA256" != "$CANDIDATE_NPUBACKEND_SHA256" ]]; then
  fail_gate "src/backend/npuBackend.cpp hash does not match the verified candidate (expected $CANDIDATE_NPUBACKEND_SHA256); apply the course patch first"
fi
NPU_HPP_SHA256="$(sha256_file "$NPU_HPP")"

# The current tracked worktree diff must byte-for-byte equal the course patch:
# any extra tracked change (or a different patch state) fails the gate. The
# same flags (--binary --no-ext-diff) are used for both sides of the compare.
TRACKED_DIFF_FILE="$(mktemp)"
if ! git -C "$MUDUO_ROOT" diff --binary --no-ext-diff > "$TRACKED_DIFF_FILE" 2>/dev/null; then
  fail_gate "cannot compute the tracked worktree diff"
fi
if ! cmp -s "$TRACKED_DIFF_FILE" "$PATCH_FILE"; then
  fail_gate "tracked worktree diff does not byte-for-byte match patch/muduoxinyu_flashattention_v1.patch; an extra tracked change or a wrong patch state was detected"
fi
TRACKED_DIFF_SHA256="$(sha256_file "$TRACKED_DIFF_FILE")"

# Freshness gate: the binary and this build log must both be at least as new
# as the newest tracked source file touched by the patch, so an obviously
# stale build artifact cannot pass. This binds the executed binary to THIS
# build's evidence; it is not claimed to be a mathematical reproducible-build
# proof, only build-source evidence binding.
PATCH_SOURCE_MTIME=0
for source in $(grep '^diff --git ' "$PATCH_FILE" | awk '{print $3}' | sed 's|^a/||'); do
  source_path="$MUDUO_ROOT/$source"
  [[ -f "$source_path" ]] || fail_gate "patched source missing: $source_path"
  source_mtime="$(stat -c %Y -- "$source_path")"
  if (( source_mtime > PATCH_SOURCE_MTIME )); then
    PATCH_SOURCE_MTIME="$source_mtime"
  fi
done
BINARY_MTIME_EPOCH="$(stat -c %Y -- "$BINARY")"
BUILD_LOG_MTIME_EPOCH="$(stat -c %Y -- "$BUILD_LOG")"
if (( BINARY_MTIME_EPOCH < PATCH_SOURCE_MTIME || BUILD_LOG_MTIME_EPOCH < PATCH_SOURCE_MTIME )); then
  fail_gate "stale build artifact: binary mtime=$BINARY_MTIME_EPOCH, build log mtime=$BUILD_LOG_MTIME_EPOCH, newest patched source mtime=$PATCH_SOURCE_MTIME"
fi
echo "AB_EVIDENCE_GATE=PASS"

ASCEND_PATH="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_PATH:-}}}"
: "${ASCEND_PATH:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}"
source "$ASCEND_PATH/set_env.sh"
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"

SMOKE_STEPS=12
PERF_STEPS=120
TIMEOUT_SECONDS=600
PROMPTS="$OUTPUT/prompts_4.txt"
printf 'hello\nhello\nhello\nhello\n' > "$PROMPTS"

# A/B 固定模型、输入和解码参数，但实现包不同：Path A=FP32 多算子，Path B=FP16 FlashAttentionV4。
COMMON=(--backend npu --enableDeviceOpt --skipValidation
        --temperature 0 --topP 1 --dumpTokenSummary)

COMMAND_LOG="$OUTPUT/commands.log"
EXIT_LOG="$OUTPUT/exit_codes.log"
: > "$COMMAND_LOG"
: > "$EXIT_LOG"

run_case() {
  local label="$1" logfile="$2"
  shift 2
  printf '%s\t' "$label" >> "$COMMAND_LOG"
  printf '%q ' timeout "$TIMEOUT_SECONDS" "$@" >> "$COMMAND_LOG"
  printf '\n' >> "$COMMAND_LOG"
  set +e
  timeout "$TIMEOUT_SECONDS" "$@" > "$logfile" 2>&1
  local rc=$?
  set -e
  printf '%s=%s\n' "$label" "$rc" >> "$EXIT_LOG"
  if [[ "$rc" -ne 0 ]]; then
    echo "RUN_CASE=FAIL label=$label exit_code=$rc log=$logfile" >&2
    return "$rc"
  fi
  echo "RUN_CASE=PASS label=$label exit_code=0 log=$logfile"
}

# 只记录与复现有关的非敏感信息，不导出完整环境变量。
npu-smi info > "$OUTPUT/device_info.txt"
if ! ATC_VERSION="$(atc --version 2>&1)"; then
  ATC_VERSION="unavailable (atc --version is unsupported)"
fi
git -C "$MUDUO_ROOT" status --porcelain --untracked-files=normal > "$OUTPUT/worktree_status.txt"
# The tracked diff was already hard-gated against the course patch above.
DIFF_SHA256="${TRACKED_DIFF_SHA256}"
MODEL_SHA256="$(sha256_file "$MODEL")"
TOKENIZER_SHA256="$(sha256_file "$TOKENIZER")"
MODEL_SIZE="$(file_size "$MODEL")"
TOKENIZER_SIZE="$(file_size "$TOKENIZER")"
BINARY_SHA256="$(sha256_file "$BINARY")"
BINARY_SIZE="$(file_size "$BINARY")"
BUILD_LOG_SHA256="$(sha256_file "$BUILD_LOG")"
BUILD_LOG_SIZE="$(file_size "$BUILD_LOG")"
# Keep auditable copies of this build's evidence inside the output directory.
cp "$BUILD_LOG" "$OUTPUT/build.log"
cp "$TRACKED_DIFF_FILE" "$OUTPUT/worktree_diff.patch"
export OUTPUT MUDUO_ROOT MODEL TOKENIZER ASCEND_PATH ATC_VERSION COMMIT DIFF_SHA256 \
  MODEL_SHA256 TOKENIZER_SHA256 MODEL_SIZE TOKENIZER_SIZE \
  BINARY BINARY_SHA256 BINARY_SIZE BINARY_MTIME_EPOCH \
  BASELINE_COMMIT PATCH_FILE PATCH_SHA256 NPU_CPP_SHA256 NPU_HPP_SHA256 \
  TRACKED_DIFF_SHA256 BUILD_LOG BUILD_LOG_SHA256 BUILD_LOG_SIZE
python3 - <<'PY'
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(os.environ["OUTPUT"])
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "muduo_root": os.environ["MUDUO_ROOT"],
    "baseline_commit": os.environ["BASELINE_COMMIT"],
    "commit": os.environ["COMMIT"],
    "worktree_diff_sha256": os.environ["DIFF_SHA256"],
    # Executed binary bound to source evidence: the patched key source
    # (npuBackend.cpp) must equal the apply_patch.sh verified candidate hash
    # and the patch file must equal the verified patch hash (AB_EVIDENCE_GATE).
    "patch": {"path": os.environ["PATCH_FILE"], "sha256": os.environ["PATCH_SHA256"]},
    "key_source_files": {
        "src/backend/npuBackend.cpp": os.environ["NPU_CPP_SHA256"],
        "src/backend/npuBackend.hpp": os.environ["NPU_HPP_SHA256"],
    },
    "binary": {
        "path": os.environ["BINARY"],
        "size": int(os.environ["BINARY_SIZE"]),
        "sha256": os.environ["BINARY_SHA256"],
        "mtime_utc": datetime.fromtimestamp(int(os.environ["BINARY_MTIME_EPOCH"]), timezone.utc).isoformat(),
    },
    "model": {
        "path": os.environ["MODEL"],
        "size": int(os.environ["MODEL_SIZE"]),
        "sha256": os.environ["MODEL_SHA256"],
    },
    "tokenizer": {
        "path": os.environ["TOKENIZER"],
        "size": int(os.environ["TOKENIZER_SIZE"]),
        "sha256": os.environ["TOKENIZER_SHA256"],
    },
    "build_evidence": {
        "build_log_path": os.environ["BUILD_LOG"],
        "build_log_sha256": os.environ["BUILD_LOG_SHA256"],
        "build_log_size": int(os.environ["BUILD_LOG_SIZE"]),
        "build_log_evidence_file": "build.log",
        "worktree_diff_sha256": os.environ["TRACKED_DIFF_SHA256"],
        "worktree_diff_file": "worktree_diff.patch",
        "evidence_gate": "PASS",
    },
    "ascend_path": os.environ["ASCEND_PATH"],
    "atc_version": os.environ["ATC_VERSION"],
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "device_info_file": "device_info.txt",
    "commands_file": "commands.log",
    "exit_codes_file": "exit_codes.log",
    "execution_order": ["smoke_a", "smoke_b", "perf_b", "perf_a"],
    "comparison_scope": "implementation package: FP32 multi-op vs FP16 FlashAttentionV4",
}
(output / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

# 冒烟顺序 A→B；带 profiling 检查路径标记、调用次数、fallback 和 token exact。
run_case smoke_a "$OUTPUT/smoke_path_a.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" hello \
  "${COMMON[@]}" --steps "$SMOKE_STEPS" \
  --profileNpu --profileNpuStart 0 --profileNpuEnd "$SMOKE_STEPS"
run_case smoke_b "$OUTPUT/smoke_path_b.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" hello \
  "${COMMON[@]}" --steps "$SMOKE_STEPS" --useNpuFlashAttention \
  --profileNpu --profileNpuStart 0 --profileNpuEnd "$SMOKE_STEPS"

# 性能顺序反转为 B→A，减小固定运行顺序偏差；sample 1 预热，sample 2~4 统计。
run_case perf_b "$OUTPUT/perf_path_b.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" "$PROMPTS" \
  "${COMMON[@]}" --steps "$PERF_STEPS" --useNpuFlashAttention
run_case perf_a "$OUTPUT/perf_path_a.log" \
  "$BINARY" "$MODEL" "$TOKENIZER" "$PROMPTS" \
  "${COMMON[@]}" --steps "$PERF_STEPS"

python3 "$(dirname "$0")/analyze_results.py" \
  --smoke-a "$OUTPUT/smoke_path_a.log" \
  --smoke-b "$OUTPUT/smoke_path_b.log" \
  --perf-a "$OUTPUT/perf_path_a.log" \
  --perf-b "$OUTPUT/perf_path_b.log" \
  --output "$OUTPUT/result.json" \
  --smoke-steps "$SMOKE_STEPS" --num-layers 12

echo "RUN_AB=PASS result=$OUTPUT/result.json manifest=$OUTPUT/run_manifest.json"
