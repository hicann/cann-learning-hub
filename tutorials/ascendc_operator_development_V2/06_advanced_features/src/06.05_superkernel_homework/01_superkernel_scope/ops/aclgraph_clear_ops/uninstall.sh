#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

custom_sk_pip_uninstall op_extension

echo "Uninstalled ACLGraph op_extension package."
