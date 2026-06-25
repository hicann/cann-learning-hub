import os
from pathlib import Path

import numpy as np


def clean_bin_files():
    for name in ("input.bin", "golden.bin", "output.bin"):
        path = Path(name)
        if path.exists():
            path.unlink()


def gen_case_data():
    shape = (8, 16, 1743)
    rng = np.random.default_rng(2)
    input_x = rng.uniform(-8.0, 8.0, size=shape).astype(np.float32)
    golden = -np.logaddexp(0.0, -input_x).astype(np.float32)
    input_x.tofile("input.bin")
    golden.tofile("golden.bin")
    print(f"generated input.bin and golden.bin, shape={shape}, dtype=float32")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    clean_bin_files()
    gen_case_data()
