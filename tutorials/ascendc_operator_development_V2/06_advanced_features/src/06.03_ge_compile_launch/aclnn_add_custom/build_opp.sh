#!/usr/bin/env bash
set -euo pipefail

normalize_host_arch() {
    case "$1" in
        x86_64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        *)
            echo "Unsupported Host architecture: $1" >&2
            return 1
            ;;
    esac
}

compute_unit_for_npu_arch() {
    case "$1" in
        dav-2201)
            echo "ascend910b"
            ;;
        dav-3510)
            echo "ascend950"
            ;;
        *)
            echo "Unsupported NPU architecture: $1 (expected dav-2201 or dav-3510)" >&2
            return 1
            ;;
    esac
}

package_filename() {
    echo "custom_opp_ubuntu_$1.run"
}

source_cann_environment() {
    local restore_nounset=false
    if [[ "$-" == *u* ]]; then
        restore_nounset=true
        set +u
    fi
    # shellcheck disable=SC1090
    source "$1"
    if [[ "$restore_nounset" == true ]]; then
        set -u
    fi
}

check_run_package() {
    local package="$1"
    local check_root="$2"
    cmake -E remove_directory "$check_root"
    cmake -E make_directory "$check_root/home" "$check_root/tmp"
    HOME="$check_root/home" TMPDIR="$check_root/tmp" "$package" --check --noexec
}

usage() {
    cat <<'EOF'
Usage: build_opp.sh [--npu-arch dav-2201|dav-3510] [--output-root PATH]

Build the AddCustom OPP for the native Host architecture. The NPU architecture
defaults to CMAKE_ASC_ARCHITECTURES, or dav-3510 when the variable is unset.
EOF
}

resolve_cann_home() {
    local candidate
    for candidate in \
        "${ASCEND_HOME_PATH:-}" \
        "${ASCEND_TOOLKIT_HOME:-}" \
        "/usr/local/Ascend/cann"; do
        if [[ -n "$candidate" && -f "$candidate/set_env.sh" ]]; then
            realpath "$candidate"
            return 0
        fi
    done
    echo "Cannot locate CANN set_env.sh; set ASCEND_HOME_PATH first." >&2
    return 1
}

main() {
    local script_dir workspace_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    workspace_root="$(realpath "$script_dir/../../..")"

    local npu_arch="${CMAKE_ASC_ARCHITECTURES:-dav-3510}"
    local output_root="$workspace_root/Sources/06.03_ge_compile_launch"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --npu-arch)
                [[ $# -ge 2 ]] || { echo "--npu-arch requires a value" >&2; return 2; }
                npu_arch="$2"
                shift 2
                ;;
            --output-root)
                [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; return 2; }
                output_root="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage >&2
                return 2
                ;;
        esac
    done

    local host_arch compute_unit cann_home build_dir package_dir output_package jobs
    host_arch="$(normalize_host_arch "$(uname -m)")"
    compute_unit="$(compute_unit_for_npu_arch "$npu_arch")"
    cann_home="$(resolve_cann_home)"
    build_dir="$(realpath -m "$output_root/opp_build/${npu_arch}-${host_arch}")"
    package_dir="$(realpath -m "$output_root/op_packages/$npu_arch")"
    output_package="$package_dir/$(package_filename "$host_arch")"
    jobs="${BUILD_JOBS:-$(nproc)}"

    source_cann_environment "$cann_home/set_env.sh"

    cmake -E remove_directory "$build_dir"
    cmake -E make_directory "$build_dir" "$package_dir"
    cmake -S "$script_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DASCEND_CANN_PACKAGE_PATH="$cann_home" \
        -DASCEND_COMPUTE_UNIT="$compute_unit" \
        -DASCEND_PYTHON_EXECUTABLE=python3 \
        -DCMAKE_INSTALL_PREFIX="$build_dir" \
        -DENABLE_BINARY_PACKAGE=True \
        -DENABLE_CROSS_COMPILE=False \
        -DENABLE_SOURCE_PACKAGE=True \
        -Dvendor_name=customize
    cmake --build "$build_dir" --target binary -j"$jobs"
    cmake --build "$build_dir" --target package -j"$jobs"

    local generated_packages=()
    while IFS= read -r package; do
        generated_packages+=("$package")
    done < <(find "$build_dir" -maxdepth 2 -type f -name 'custom_opp*.run' | sort)
    if [[ ${#generated_packages[@]} -ne 1 ]]; then
        echo "Expected one generated RUN package, found ${#generated_packages[@]} in $build_dir" >&2
        return 1
    fi

    install -m 0755 "${generated_packages[0]}" "$output_package"
    check_run_package "$output_package" "$build_dir/package_check"

    echo "Host architecture: $host_arch"
    echo "NPU architecture: $npu_arch"
    echo "Compute unit: $compute_unit"
    echo "OPP_PACKAGE=$output_package"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
