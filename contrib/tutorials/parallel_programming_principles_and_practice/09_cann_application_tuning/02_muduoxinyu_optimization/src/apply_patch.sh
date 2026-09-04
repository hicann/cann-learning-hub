#!/usr/bin/env bash
# MuduoXinyu 课程：安全补丁应用脚本（离线）。
#
# 前置校验（全部通过才允许应用补丁）：
#   2. 目标是 git 仓库
#   2. 仓库 HEAD 必须是基线提交 60c6371cd30894d9896dfa979b86c6f892b6cbda
#   3. 补丁文件存在，且 SHA256 必须等于
#      ecb725364fb9ecfae8ee1b4530890438af88164c244fd730b778a19773e6223a
#      （哈希不匹配即失败，不存在跳过哈希的能力）
#   4. 工作树必须干净（含未跟踪文件，git status --porcelain 为空）；若补丁已应用，明确拒绝重复应用
#   5. `git apply --check` 通过
# 应用方式：仅 `git apply`，绝不执行回退（reset）、清理（clean）、切换（checkout）类命令。
# 应用后校验：src/backend/npuBackend.cpp 的 SHA256 必须等于
#   3dbe660a5933584fc9d434100ea4dc5c1600186a494b46df96da6aea55612126（强制，不可跳过）。
#
# 用法：
#   bash apply_patch.sh --muduo-root <repo> [--patch <file>] [--check-only]
#
# 本脚本不联网、不 reset、不 clean、不 checkout。

set -euo pipefail

BASELINE_COMMIT="60c6371cd30894d9896dfa979b86c6f892b6cbda"
PATCH_SHA256="ecb725364fb9ecfae8ee1b4530890438af88164c244fd730b778a19773e6223a"
CANDIDATE_NPUBACKEND_SHA256="3dbe660a5933584fc9d434100ea4dc5c1600186a494b46df96da6aea55612126"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PATCH="$SCRIPT_DIR/patch/muduoxinyu_flashattention_v1.patch"

MUDUO_ROOT=""
PATCH_PATH="$DEFAULT_PATCH"
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash apply_patch.sh --muduo-root <dir> [--patch <file>] [--check-only]
补丁 SHA256 与应用后 npuBackend.cpp 的 SHA256 均为强制校验，不可跳过。
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --muduo-root) MUDUO_ROOT="${2:-}"; shift 2 ;;
        --patch) PATCH_PATH="${2:-}"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --help|-h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$MUDUO_ROOT" ]]; then
    usage
fi

fail() {
    echo "PATCH_APPLY=FAIL $*" >&2
    exit 2
}

sha256_file() {
    # GNU sha256sum prefixes a backslash when an input filename needs escaping.
    sha256sum -- "$1" | sed 's/^\\//' | awk '{print $1}'
}

# ---------- 1. git 仓库（兼容普通 clone 与 linked worktree） ----------
if ! git -C "$MUDUO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    fail "not a git repository: $MUDUO_ROOT"
fi
echo "REPO_GIT=PASS"

# ---------- 2. 基线提交 ----------
CURRENT_COMMIT="$(git -C "$MUDUO_ROOT" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$BASELINE_COMMIT" ]]; then
    fail "HEAD is $CURRENT_COMMIT, expected baseline $BASELINE_COMMIT"
fi
echo "REPO_COMMIT=PASS ($CURRENT_COMMIT)"

# ---------- 3. 补丁存在 + SHA256 强制校验 ----------
if [[ ! -f "$PATCH_PATH" ]]; then
    fail "patch file not found: $PATCH_PATH"
fi
echo "PATCH_FILE=PASS"
ACTUAL_PATCH_SHA256="$(sha256_file "$PATCH_PATH")"
if [[ "$ACTUAL_PATCH_SHA256" != "$PATCH_SHA256" ]]; then
    echo "PATCH_SHA256=FAIL" >&2
    echo "  actual:   $ACTUAL_PATCH_SHA256" >&2
    echo "  expected: $PATCH_SHA256" >&2
    fail "patch hash mismatch; refusing to apply (no skip-hash option)"
fi
echo "PATCH_SHA256=PASS ($ACTUAL_PATCH_SHA256)"

# ---------- 4. 干净工作树 / 重复应用识别 ----------
WORKTREE_STATUS="$(git -C "$MUDUO_ROOT" status --porcelain --untracked-files=normal)"
if [[ -n "$WORKTREE_STATUS" ]]; then
    if git -C "$MUDUO_ROOT" apply --reverse --check "$PATCH_PATH" 2>/dev/null; then
        fail "patch appears to be already applied; refusing to reapply"
    fi
    echo "WORKTREE_DIRTY=FAIL" >&2
    printf '%s\n' "$WORKTREE_STATUS" >&2
    fail "working tree is not clean; apply_patch never resets or discards changes"
fi
echo "WORKTREE_CLEAN=PASS"

# ---------- 5. git apply --check ----------
if ! git -C "$MUDUO_ROOT" apply --check "$PATCH_PATH"; then
    fail "git apply --check failed; patch does not match the baseline working tree"
fi
echo "PATCH_CHECK=PASS"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "PATCH_CHECK_ONLY=PASS"
    exit 0
fi

# ---------- 6. 应用 ----------
git -C "$MUDUO_ROOT" apply "$PATCH_PATH"
echo "PATCH_APPLY_CMD=git apply $PATCH_PATH"

# ---------- 7. 应用后关键文件哈希校验（强制） ----------
NPU_CPP="$MUDUO_ROOT/src/backend/npuBackend.cpp"
if [[ ! -f "$NPU_CPP" ]]; then
    echo "FILE_SHA256_CHECK=FAIL" >&2
    fail "src/backend/npuBackend.cpp not found after patch"
fi
ACTUAL_SHA256="$(sha256_file "$NPU_CPP")"
if [[ "$ACTUAL_SHA256" != "$CANDIDATE_NPUBACKEND_SHA256" ]]; then
    echo "FILE_SHA256_CHECK=FAIL" >&2
    echo "  actual:   $ACTUAL_SHA256" >&2
    echo "  expected: $CANDIDATE_NPUBACKEND_SHA256" >&2
    fail "post-apply hash mismatch; repository or patch drifted from the verified candidate"
fi
echo "FILE_SHA256_CHECK=PASS"

echo "PATCH_APPLY=PASS"
echo "CHANGED_FILES:"
git -C "$MUDUO_ROOT" diff --stat
