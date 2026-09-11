#!/bin/bash

custom_sk_python() {
    echo "${PYTHON:-python3}"
}

custom_sk_build_jobs() {
    if [[ -n "${SK_BUILD_JOBS:-}" ]]; then
        echo "${SK_BUILD_JOBS}"
    else
        nproc
    fi
}

custom_sk_require_command() {
    local command_name=$1
    local hint=${2:-}
    if command -v "${command_name}" >/dev/null 2>&1; then
        return
    fi

    if [[ -n "${hint}" ]]; then
        echo "${command_name} command not found. ${hint}" >&2
    else
        echo "${command_name} command not found." >&2
    fi
    exit 1
}

custom_sk_pip_install() {
    "$(custom_sk_python)" -m pip install --no-build-isolation --force-reinstall "$@"
}

custom_sk_pip_uninstall() {
    "$(custom_sk_python)" -m pip uninstall -y "$@"
}

custom_sk_reset_dir() {
    local target_dir=$1
    if [[ -z "${target_dir}" || "${target_dir}" == "/" || "${target_dir}" == "." ]]; then
        echo "Refuse to reset unsafe directory: ${target_dir}" >&2
        exit 1
    fi

    rm -rf "${target_dir}"
    mkdir -p "${target_dir}"
}
