#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m pip install --user --no-index --no-deps --find-links "$ROOT/wheelhouse" \
    "transformers==4.53.2" \
    "tokenizers==0.21.4" \
    "huggingface-hub==0.36.2" \
    "safetensors==0.8.0" \
    "einops==0.8.1" \
    "regex==2026.7.19" \
    "tqdm==4.70.0"

echo "[PASS] offline Qwen dependencies installed"
