import os
from pathlib import Path

import numpy as np


def clean_bin_files():
    for name in ("input.bin", "golden.bin", "output.bin"):
        path = Path(name)
        if path.exists():
            path.unlink()


def gen_case_data():
    shape = (4, 2028)
    rng = np.random.default_rng(3)
    input_x = rng.uniform(-8.0, 8.0, size=shape).astype(np.float16)
    sigmoid = np.float16(1.0) / (np.float16(1.0) + np.exp(-input_x))
    golden = np.log(sigmoid)
    input_x.tofile("input.bin")
    golden.tofile("golden.bin")
    print(f"generated input.bin and golden.bin, shape={shape}, dtype=float16")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    clean_bin_files()
    gen_case_data()
