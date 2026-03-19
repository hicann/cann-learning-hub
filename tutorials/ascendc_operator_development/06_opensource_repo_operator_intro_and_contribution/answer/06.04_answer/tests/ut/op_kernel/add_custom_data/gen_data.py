import numpy as np
import os

def gen_golden_data_simple():

    input_x = np.random.uniform(1, 100, [2, 1024]).astype(np.float32)
    input_y = np.random.uniform(1, 100, [2, 1024]).astype(np.float32)
    golden = input_x + input_y
    input_x.tofile("./input1.bin")
    input_y.tofile("./input2.bin")

    golden.tofile("./golden.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_golden_data_simple()