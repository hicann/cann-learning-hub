import os
import numpy as np
import ml_dtypes

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def gen_golden_data():
    m, n, k, is_bias = 128, 128, 128, True

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x1_gm = np.random.randint(-100, 100, [m, k]).astype(np.float16)
    x2_gm = np.random.randint(-100, 100, [k, n]).astype(np.float16)

    if is_bias:
        bias_gm = np.random.randint(-10, 10, [n]).reshape([n]).astype(np.float16)
        matmul_golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32) + bias_gm
        bias_gm.tofile("./input/bias_gm.bin")
    else:
        matmul_golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)

    mmad_leakyrelu_golden = leaky_relu(matmul_golden, alpha=0.001)

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    mmad_leakyrelu_golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data()
