import numpy as np
import os

def gen_golden_data_simple():
    input_x = np.random.uniform(1, 100, [8, 2048]).astype(np.float32)
    golden = 1/ (1 + np.exp(-input_x))
    input_x.tofile("./input.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_golden_data_simple()