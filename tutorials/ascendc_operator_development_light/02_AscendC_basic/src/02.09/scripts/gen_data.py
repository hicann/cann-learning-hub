import os
import numpy as np
import argparse

np.random.seed(42)
def gen_golden_data(shape, case_num, dtype):
    if dtype == "float":
        np_dtype = np.float32
    elif dtype == "float16":
        np_dtype = np.float16
    src_gm = np.random.randint(-5, 5, shape).astype(np_dtype)
    golden = 1 / (1 + np.exp(-1 * src_gm))
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    src_gm.tofile("./input/src_gm_case{}.bin".format(case_num))
    golden.tofile("./output/golden_case{}.bin".format(case_num))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # nargs='*' 表示接收 0 个或多个参数，解析后自动存入列表
    parser.add_argument('--shape', nargs='*', type=int, default=[])
    parser.add_argument('--case-num', type=int, default=0)
    parser.add_argument('--dtype', type=str, default='float')
    args = parser.parse_args()
    gen_golden_data(args.shape, args.case_num, args.dtype)