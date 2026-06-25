import sys
import os
from pathlib import Path

import numpy as np


def verify_result():
    golden = np.fromfile("golden.bin", dtype=np.float16).astype(np.float32)
    output = np.fromfile("output.bin", dtype=np.float16).astype(np.float32)
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
