# ============================================================================
# 测试数据生成脚本 - rope_optimized
# ============================================================================
#
# 默认生成 Level 0 测试数据（B=2, S=8, H=2, D=64, FP32）
# 可通过命令行参数覆盖

import numpy as np
import os
import sys
import math

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 默认参数：Level 0 小规模数据
B = 2
S = 8
H = 2
D = 64
DTYPE = np.float32

# 命令行参数覆盖
if len(sys.argv) >= 5:
    B = int(sys.argv[1])
    S = int(sys.argv[2])
    H = int(sys.argv[3])
    D = int(sys.argv[4])
if len(sys.argv) >= 6:
    dtype_str = sys.argv[5].lower()
    if dtype_str in ("half", "fp16", "float16"):
        DTYPE = np.float16
    elif dtype_str in ("float", "fp32", "float32"):
        DTYPE = np.float32

# 设置随机种子保证可复现
np.random.seed(42)

# 生成 x: [B, S, H, D]
x = np.random.randn(B, S, H, D).astype(DTYPE)

# 生成 cos/sin: [S, D]
# 使用标准 RoPE 位置编码公式
position = np.arange(S, dtype=np.float32).reshape(-1, 1)  # [S, 1]
div_term = np.exp(np.arange(0, D, 2, dtype=np.float32) * (-math.log(10000.0) / D))  # [D/2]
angles = position * div_term  # [S, D/2]

cos = np.zeros((S, D), dtype=DTYPE)
sin = np.zeros((S, D), dtype=DTYPE)
# halves 布局：后半维复制前半维的取值（与 kernel 的 rotate_half 配套，见正文 11.1.1）
cos[:, : D // 2] = np.cos(angles).astype(DTYPE)
cos[:, D // 2 :] = np.cos(angles).astype(DTYPE)
sin[:, : D // 2] = np.sin(angles).astype(DTYPE)
sin[:, D // 2 :] = np.sin(angles).astype(DTYPE)

# 保存输入
x.tofile("input/input_x.bin")
cos.tofile("input/input_cos.bin")
sin.tofile("input/input_sin.bin")

# 计算 golden
golden = compute_golden(x, cos, sin)
golden.tofile("output/golden.bin")

print(f"Generated test data: B={B}, S={S}, H={H}, D={D}, dtype={DTYPE}")
print(f"  input/input_x.bin:   shape={x.shape}, dtype={x.dtype}, bytes={x.nbytes}")
print(f"  input/input_cos.bin: shape={cos.shape}, dtype={cos.dtype}, bytes={cos.nbytes}")
print(f"  input/input_sin.bin: shape={sin.shape}, dtype={sin.dtype}, bytes={sin.nbytes}")
print(f"  output/golden.bin:   shape={golden.shape}, dtype={golden.dtype}, bytes={golden.nbytes}")
