#!/usr/bin/env bash
# QwenRoPeCustom profiling 脚本
# 使用 CANN msprof 工具分析 kernel 瓶颈
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

TOTAL_TOKENS="${TOTAL_TOKENS:-128}"
HEAD_DIM="${HEAD_DIM:-64}"
BLOCK_DIM="${BLOCK_DIM:-8}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-10}"
MSPROF="${MSPROF:-msprof}"
AIC_METRICS="${AIC_METRICS:-Memory}"
PROFILE_NAME="${PROFILE_NAME:-tokens_${TOTAL_TOKENS}_hd_${HEAD_DIM}_bd_${BLOCK_DIM}}"

BINARY="$ROOT_DIR/out/bin/rope_baseline_standalone"
if [[ ! -f "$BINARY" ]]; then
    echo "[ERROR] Binary not found. Run scripts/build.sh first."
    exit 1
fi

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d "$HOME/Ascend/ascend-toolkit/cann-8.5.0" ]]; then
    CANN_PATH="$HOME/Ascend/ascend-toolkit/cann-8.5.0"
elif [[ -d "/usr/local/Ascend/ascend-toolkit/latest" ]]; then
    CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"
else
    CANN_PATH=""
fi

export LD_LIBRARY_PATH="$ROOT_DIR/out/lib${CANN_PATH:+:$CANN_PATH/lib64}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── 生成 profiling 输入 (含 cos/sin) ───────────────────────
echo "[PROFILE] Generating input data..."
cd "$ROOT_DIR"
mkdir -p input output
python3 -c "
import numpy as np
rng = np.random.default_rng(42)
tt, hd = $TOTAL_TOKENS, $HEAD_DIM
x   = rng.normal(0, 1, size=(tt, hd)).astype(np.float32)
# Qwen2.5 RoPE cos/sin (base=1e6)
inv_freq = 1.0 / (1000000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
freqs = np.outer(np.arange(tt, dtype=np.float64), inv_freq)
emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)
cos = np.cos(emb)
sin = np.sin(emb)
x.tofile('input/input_x.bin')
cos.tofile('input/input_cos.bin')
sin.tofile('input/input_sin.bin')
print(f'[OK] input_x/cos/sin.bin tokens=$TOTAL_TOKENS head_dim=$HEAD_DIM')
"

APP_WRAPPER="$ROOT_DIR/output/profile/run_rope_profile_${PROFILE_NAME}.sh"
mkdir -p "$(dirname "$APP_WRAPPER")"
cat > "$APP_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
exec "$BINARY" --tokens "$TOTAL_TOKENS" --head-dim "$HEAD_DIM" --block-dim "$BLOCK_DIM" --warmup "$WARMUP" --repeat "$REPEAT" --rounds 1
EOF
chmod +x "$APP_WRAPPER"

# ── 运行 msprof ────────────────────────────────────────────
echo "[PROFILE] Running msprof..."
"$MSPROF" --application="$APP_WRAPPER" \
    --output="$ROOT_DIR/output/profile/$PROFILE_NAME" \
    --ascendcl=on \
    --runtime-api=on \
    --task-time=on \
    --ai-core=on \
    --aic-metrics="$AIC_METRICS" \
    --task-memory=on \
    --sys-hardware-mem=on \
    --sys-io-profiling=on

echo "[PROFILE] Done. Output: output/profile/$PROFILE_NAME"
echo "[PROFILE] Parse with: $MSPROF --parse=on --output=output/profile/$PROFILE_NAME"
