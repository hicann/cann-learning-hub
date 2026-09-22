#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

custom_sk_require_command bisheng "Please source CANN set_env.sh first."

custom_sk_pip_install "${SCRIPT_DIR}"

echo "Installed ACLGraph op_extension package from ${SCRIPT_DIR}."
