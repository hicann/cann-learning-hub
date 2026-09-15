import os

import numpy as np
import ml_dtypes
import en_dtypes

bfloat16 = ml_dtypes.bfloat16
fp4_e1m2x2 = en_dtypes.float4_e1m2


def pack_two_fp4(scale_matrix):
    scale_matrix_row = scale_matrix.shape[0]
    scale_matrix_col = scale_matrix.shape[1]
    scale_matrix_bin = scale_matrix.flatten()
    scale_matrix_high = scale_matrix_bin[::2].view(np.uint8)
    scale_matrix_low = scale_matrix_bin[1::2].view(np.uint8)
    low_bits = (scale_matrix_low & 0x0F) << 4
    high_bits = scale_matrix_high & 0x0F
    combined = low_bits | high_bits
    scale_matrix_bin = combined.reshape(scale_matrix_row, scale_matrix_col // 2)
    return scale_matrix_bin


def main():
    m, n, k = 8192, 8192, 8192
    sk = (int)(np.ceil(k / 64) * 2)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    np.random.seed(42)
    x1_gm = np.random.uniform(-1, 2, [m, k]).astype(fp4_e1m2x2)
    x2_gm = np.random.uniform(-1, 2, [k, n]).astype(fp4_e1m2x2)

    x1_scale_gm = np.random.randint(127, 130, [m, sk]).astype(np.uint8)
    x2_scale_gm = np.random.randint(127, 130, [sk, n]).astype(np.uint8)

    x1_mx = 2 ** (x1_scale_gm.astype(np.float64) - 127)
    x2_mx = 2 ** (x2_scale_gm.astype(np.float64) - 127)
    x1_full = np.zeros([m, k], dtype=np.float64)
    x2_full = np.zeros([k, n], dtype=np.float64)

    for i in range(x1_gm.shape[1]):
        x1_full[:, i] = x1_gm[:, i] * x1_mx[:, i // 32]
        x2_full[i, :] = x2_gm[i, :] * x2_mx[i // 32, :]

    golden = np.matmul(x1_full.astype(np.float64), x2_full.astype(np.float64)).astype(bfloat16)

    x2_gm = x2_gm.transpose()
    x2_scale_gm = x2_scale_gm.transpose()
    x1_gm_packed = pack_two_fp4(x1_gm)
    x2_gm_packed = pack_two_fp4(x2_gm)
    x1_gm_packed.tofile("input/x1_gm.bin")
    x2_gm_packed.tofile("input/x2_gm.bin")
    x1_scale_gm.tofile("input/x1_scale_gm.bin")
    x2_scale_gm.tofile("input/x2_scale_gm.bin")
    golden.tofile("output/golden.bin")

    print(f"generated MxFP4 input data: A[{m},{k}], B[{k},{n}], ScaleA[{m},{sk}], ScaleB[{sk},{n}]")
    print(f"golden output saved to output/golden.bin")


if __name__ == "__main__":
    main()
