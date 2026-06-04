import sys
import numpy as np
import ml_dtypes

RELATIVE_TOL = 1e-3
ABSOLUTE_TOL = 1e-3
ERROR_TOL = 1e-3


def verify_result(output, golden):
    output = np.fromfile(output, dtype=np.float16).reshape(-1)
    golden = np.fromfile(golden, dtype=np.float16).reshape(-1)

    # Get total number of elements compared
    total_elements = golden.size

    different_element_results = np.isclose(output.astype(np.float32),
                                           golden.astype(np.float32),
                                           rtol=RELATIVE_TOL,
                                           atol=ABSOLUTE_TOL,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]

    # Get total number of errors
    error_count = different_element_indexes.size

    # Print total comparison count and error count
    print(f"Total elements compared: {total_elements}")
    print(f"Total error elements: {error_count}")

    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = float(golden[real_index])
        output_data = float(output[real_index])
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
            abs(output_data - golden_data) / golden_data))
        if index == 100:
            break

    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == '__main__':
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass!")
    except Exception as e:
        print(e)
        sys.exit(1)
