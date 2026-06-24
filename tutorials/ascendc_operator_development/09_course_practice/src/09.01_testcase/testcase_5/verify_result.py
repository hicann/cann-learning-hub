import sys
import os
from pathlib import Path

import numpy as np


def bfloat16_bits_to_float32(data):
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def verify_result():
    golden = bfloat16_bits_to_float32(np.fromfile("golden.bin", dtype=np.uint16))
    output = bfloat16_bits_to_float32(np.fromfile("output.bin", dtype=np.uint16))
    if golden.size != output.size:
        print(f"FAILED! size mismatch, output={output.size}, golden={golden.size}")
        return False
    diff = np.abs(output - golden)
    close = np.isclose(output, golden, rtol=1e-3, atol=1e-3, equal_nan=True)
    diff_idx = np.where(close != True)[0]
    if diff_idx.size == 0:
        print("TEST PASSED!")
        return True
    print("TEST FAILED!")
    print(f"mismatch count: {diff_idx.size}/{golden.size}")
    print(f"max abs diff: {diff.max() if diff.size else 0.0}")
    for idx in diff_idx[:10]:
        print(f"index: {idx}, output: {output[idx]}, golden: {golden[idx]}, diff: {diff[idx]}")
    return False


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    ok = verify_result()
    sys.exit(0 if ok else 1)
