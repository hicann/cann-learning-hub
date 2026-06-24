import os
from pathlib import Path

import numpy as np


def float32_to_bfloat16_bits(data):
    bits = np.asarray(data, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bfloat16_bits_to_float32(data):
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def clean_bin_files():
    for name in ("input.bin", "golden.bin", "output.bin"):
        path = Path(name)
        if path.exists():
            path.unlink()


def gen_case_data():
    shape = (64, 64, 1024)
    rng = np.random.default_rng(6)
    input_x_bits = float32_to_bfloat16_bits(rng.uniform(-8.0, 8.0, size=shape).astype(np.float32))
    input_x = bfloat16_bits_to_float32(input_x_bits)
    sigmoid = np.float32(1.0) / (np.float32(1.0) + np.exp(-input_x, dtype=np.float32))
    golden = np.log(sigmoid, dtype=np.float32)
    input_x_bits.tofile("input.bin")
    float32_to_bfloat16_bits(golden).tofile("golden.bin")
    print(f"generated input.bin and golden.bin, shape={shape}, dtype=bfloat16")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    clean_bin_files()
    gen_case_data()
