import os

import numpy as np


M = 8192
N = 8192
K = 8192


def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    np.random.seed(0)
    a = np.random.uniform(-1, 1, (M, K)).astype(np.float16)
    b = np.random.uniform(-1, 1, (K, N)).astype(np.float16)

    a.tofile("input/input_x.bin")
    b.T.copy().tofile("input/input_y.bin")

    golden = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    golden.tofile("output/golden.bin")

    print(f"generated A[{M},{K}] and transposed B file for logical B[{K},{N}]")
    print(f"golden output saved to output/golden.bin")


if __name__ == "__main__":
    main()
