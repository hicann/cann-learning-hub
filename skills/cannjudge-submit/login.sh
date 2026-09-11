#!/usr/bin/env bash
# Run with: bash /path/to/cannjudge-submit/login.sh --account your@email.com
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "${SCRIPT_DIR}/cannjudge_cli.py" login --captcha-display file "$@"
