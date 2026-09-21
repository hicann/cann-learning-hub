import os
import numpy as np
import argparse

np.random.seed(42)
def gen_golden_data():
    src0 = np.random.randint(-5, 5, [33, 31]).astype(np.float16)
    src1 = np.random.randint(-5, 5, [33, 31]).astype(np.float16)
    golden = src0 + src1
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    src0.tofile("./input/src0.bin")
    src1.tofile("./input/src1.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data()