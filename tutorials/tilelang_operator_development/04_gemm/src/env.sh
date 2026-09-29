#!/bin/bash
# Source before running Python or Jupyter.
_COURSE_PREVIOUS_SRC="${COURSE_ROOT:+$COURSE_ROOT/src}"
COURSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CANN_ENV_SCRIPT:-/usr/local/Ascend/set_env.sh}" || return 1
# Optional: select an existing, compiled course-compatible TileLang checkout.
# If unset, use the TileLang package installed in the current Python environment.
if [ -n "${TILELANG_COURSE_FORK:-}" ]; then
    if [ ! -f "$TILELANG_COURSE_FORK/tilelang/__init__.py" ]; then
        echo "TileLang fork not found: $TILELANG_COURSE_FORK" >&2
        return 1
    fi
    export TILELANG_COURSE_FORK
fi
# Remove the previous chapter's source directory when switching chapters.
_COURSE_PYTHONPATH=""
IFS=: read -r -a _COURSE_PATH_PARTS <<< "${PYTHONPATH:-}"
for _COURSE_PATH_PART in "${_COURSE_PATH_PARTS[@]}"; do
    if [ -n "$_COURSE_PATH_PART" ] && [ "$_COURSE_PATH_PART" != "$_COURSE_PREVIOUS_SRC" ] && [ "$_COURSE_PATH_PART" != "$COURSE_ROOT/src" ]; then
        _COURSE_PYTHONPATH="${_COURSE_PYTHONPATH:+$_COURSE_PYTHONPATH:}$_COURSE_PATH_PART"
    fi
done
export PYTHONPATH="$COURSE_ROOT/src${_COURSE_PYTHONPATH:+:$_COURSE_PYTHONPATH}"
# Follow the chapter when the previous output path was selected automatically.
if [ -z "${COURSE_OUTPUT_DIR:-}" ] || [ "${COURSE_OUTPUT_DIR:-}" = "${_TILELANG_COURSE_AUTO_OUTPUT:-}" ]; then
    unset COURSE_OUTPUT_DIR
    _COURSE_AUTO_OUTPUT=1
else
    _COURSE_AUTO_OUTPUT=0
fi
export COURSE_OUTPUT_DIR="${COURSE_OUTPUT_DIR:-$(python -c 'import os,tempfile; from pathlib import Path; print(Path(tempfile.gettempdir()) / ("tilelang_course_" + str(os.getuid())) / "04_gemm")')}"
if [ "$_COURSE_AUTO_OUTPUT" = 1 ]; then
    export _TILELANG_COURSE_AUTO_OUTPUT="$COURSE_OUTPUT_DIR"
else
    unset _TILELANG_COURSE_AUTO_OUTPUT
fi
export TILELANG_CACHE_DIR="$COURSE_OUTPUT_DIR/tilelang_cache"
export TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TILELANG_CACHE_DIR"
