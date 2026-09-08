#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$ROOT/qwen2_5_five_ops_benchmark.py" "$@"
