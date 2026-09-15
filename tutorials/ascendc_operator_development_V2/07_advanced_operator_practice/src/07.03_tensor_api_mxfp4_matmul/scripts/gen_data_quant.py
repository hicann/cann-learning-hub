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

    golden_f64 = np.matmul(x1_full.astype(np.float64), x2_full.astype(np.float64))

    # 量化参数：per-channel 量化 (F32 -> INT8)
    quant_scale = np.ones([1, n], dtype=np.float32) * 0.01
    quant_offset = np.zeros([1, n], dtype=np.float32)

    # Fixpipe 量化参数为 uint64 打包格式（参考 SetQuantVector 接口的参数格式）：
    #   quant = (scale & 0xFFFFE000) | (1 << 46) | ((offset & 0x1FF) << 37)
    # scale 按 float32 位截取高 19 位（1 符号 + 8 指数 + 10 尾数），bit46 为标志位；
    # offset 四舍五入取整并截断到 [-256, 255]，按 9 位存于 bit37~45。
    scale_bits = np.frombuffer(quant_scale, np.uint32).reshape(1, n) & np.uint32(0xFFFFE000)
    scale_hw = scale_bits.view(np.float32)  # 硬件实际生效的截断后 scale
    offset_int = np.clip(np.round(quant_offset), -256, 255).astype(np.int64)
    quant_scale_tensor = scale_bits.astype(np.uint64) | (np.uint64(1) << np.uint64(46))
    quant_scale_tensor |= (offset_int.astype(np.uint64) & np.uint64(0x1FF)) << np.uint64(37)
    quant_offset_tensor = quant_scale_tensor.copy()  # offset 已打包进 quant 参数，kernel 当前仅使用 scale

    # INT8 golden = clip(round(golden_f64 * scale + offset), -128, 127)
    # golden 使用截断后的 scale 与取整后的 offset，保证与硬件 Fixpipe 量化语义一致
    golden_quant = np.clip(
        np.round(golden_f64 * scale_hw + offset_int), -128, 127
    ).astype(np.int8)

    x2_gm = x2_gm.transpose()
    x2_scale_gm = x2_scale_gm.transpose()
    x1_gm_packed = pack_two_fp4(x1_gm)
    x2_gm_packed = pack_two_fp4(x2_gm)
    x1_gm_packed.tofile("input/x1_gm.bin")
    x2_gm_packed.tofile("input/x2_gm.bin")
    x1_scale_gm.tofile("input/x1_scale_gm.bin")
    x2_scale_gm.tofile("input/x2_scale_gm.bin")
    quant_scale_tensor.tofile("input/quant_scale.bin")
    quant_offset_tensor.tofile("input/quant_offset.bin")
    golden_quant.tofile("output/golden_quant.bin")

    print(f"generated MxFP4 input data: A[{m},{k}], B[{k},{n}], ScaleA[{m},{sk}], ScaleB[{sk},{n}]")
    print(f"generated quant params (uint64 packed): Scale[1,{n}], Offset[1,{n}]")
    print(f"golden INT8 output saved to output/golden_quant.bin")


if __name__ == "__main__":
    main()
