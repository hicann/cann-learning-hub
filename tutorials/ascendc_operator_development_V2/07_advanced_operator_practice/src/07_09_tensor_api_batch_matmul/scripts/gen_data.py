import os

import numpy as np

B = 128
M = 32
K = 32
N = 32


def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    np.random.seed(42)
    # BatchMatmul：C[b] = A[b] * B[b] + Bias[b]，三路输入均为 half、ND 布局（不转置）
    a = np.random.uniform(1, 10, (B, M, K)).astype(np.float16)
    b = np.random.uniform(1, 10, (B, K, N)).astype(np.float16)
    bias = np.random.uniform(1, 10, (B, 1, N)).astype(np.float16)

    a.tofile("input/x1_gm.bin")
    b.tofile("input/x2_gm.bin")
    bias.tofile("input/bias.bin")

    # golden：FP32 计算矩阵乘加 Bias 后转 half（与硬件 L0C FP32 累加 → Fixpipe 转 half 一致）
    golden = np.matmul(a.astype(np.float32), b.astype(np.float32)) + bias.astype(np.float32)
    golden.astype(np.float16).tofile("output/golden.bin")

    print(f"generated BatchMatmul inputs: A[{B},{M},{K}], B[{B},{K},{N}], Bias[{B},1,{N}] (half, ND layout)")
    print(f"golden output saved to output/golden.bin")


if __name__ == "__main__":
    main()
