import sys
import numpy as np

RELATIVE_TOL = 1e-6
ABSOLUTE_TOL = 1e-9
ERROR_TOL = 1e-4

def verify_result(output, golden, np_dtype):
    output = np.fromfile(output, dtype=np_dtype).reshape(-1)
    golden = np.fromfile(golden, dtype=np_dtype).reshape(-1)
    different_element_results = np.isclose(output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL, equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
            abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    print("golden_data : ", golden)
    print("output : ", output)
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == '__main__':
    try:
        if sys.argv[3] == "float":
            np_dtype = np.float32
        else:
            np_dtype = np.float16
        print("========= Compare {} and {} start =========".format(sys.argv[1], sys.argv[2]))
        res = verify_result(sys.argv[1], sys.argv[2], np_dtype)
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass!")
        print("========= Compare {} and {} end   =========".format(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e)
        sys.exit(1)
