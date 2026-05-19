"""Test data generation and verification for as_strided operator."""
import os
import sys
import numpy as np

# Add scripts directory to path for golden import
sys.path.insert(0, os.path.dirname(__file__))
from golden import as_strided_golden


# ============== Test Cases ==============

TEST_CASES = [
    # T1: Basic positive stride 1D
    {
        "name": "T1_basic_pos_stride_1d_fp32",
        "input_shape": [10],
        "size": [3],
        "stride": [2],
        "storage_offset": 1,
        "dtype": np.float32,
    },
    # T2: Basic positive stride 2D
    {
        "name": "T2_basic_pos_stride_2d_fp32",
        "input_shape": [12],
        "size": [2, 3],
        "stride": [3, 1],
        "storage_offset": 0,
        "dtype": np.float32,
    },
    # T3: Negative stride
    {
        "name": "T3_neg_stride_fp32",
        "input_shape": [10],
        "size": [3],
        "stride": [-1],
        "storage_offset": 9,
        "dtype": np.float32,
    },
    # T4: Zero stride
    {
        "name": "T4_zero_stride_fp32",
        "input_shape": [10],
        "size": [3],
        "stride": [0],
        "storage_offset": 5,
        "dtype": np.float32,
    },
    # T5: Mixed stride 3D
    {
        "name": "T5_mixed_stride_3d_fp32",
        "input_shape": [24],
        "size": [2, 2, 3],
        "stride": [6, 3, 1],
        "storage_offset": 0,
        "dtype": np.float32,
    },
    # T6: Non-zero offset fp16
    {
        "name": "T6_nonzero_offset_fp16",
        "input_shape": [20],
        "size": [3],
        "stride": [2],
        "storage_offset": 3,
        "dtype": np.float16,
    },
    # T7: Non-aligned 4D fp16
    {
        "name": "T7_4d_fp16",
        "input_shape": [64],
        "size": [2, 2, 2, 2],
        "stride": [8, 4, 2, 1],
        "storage_offset": 0,
        "dtype": np.float16,
    },
    # T8: int32 basic
    {
        "name": "T8_basic_int32",
        "input_shape": [10],
        "size": [3],
        "stride": [2],
        "storage_offset": 1,
        "dtype": np.int32,
    },
    # T9: Large shape Path A
    {
        "name": "T9_large_pathA_fp32",
        "input_shape": [50000],
        "size": [10000],
        "stride": [3],
        "storage_offset": 0,
        "dtype": np.float32,
    },
    # T10: Boundary offset
    {
        "name": "T10_boundary_offset_fp32",
        "input_shape": [10],
        "size": [1],
        "stride": [1],
        "storage_offset": 9,
        "dtype": np.float32,
    },
]


def gen_test_data(test_case, output_dir):
    """Generate binary test data for a single test case."""
    name = test_case["name"]
    input_shape = test_case["input_shape"]
    size = test_case["size"]
    stride = test_case["stride"]
    storage_offset = test_case["storage_offset"]
    dtype = test_case["dtype"]

    # Generate input data
    if dtype == np.float32:
        input_x = np.arange(np.prod(input_shape), dtype=np.float32)
    elif dtype == np.float16:
        input_x = np.arange(np.prod(input_shape), dtype=np.float32).astype(np.float16)
    elif dtype == np.int32:
        input_x = np.arange(np.prod(input_shape), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    input_x = input_x.reshape(input_shape)

    # Compute golden output
    output = as_strided_golden(input_x, size, stride, storage_offset)

    # Save binary data
    case_dir = os.path.join(output_dir, name)
    os.makedirs(case_dir, exist_ok=True)

    input_x.flatten().tofile(os.path.join(case_dir, "input_x.bin"))
    output.flatten().tofile(os.path.join(case_dir, "output_golden.bin"))

    # Save metadata
    with open(os.path.join(case_dir, "meta.txt"), "w") as f:
        f.write(f"dtype={dtype.__name__}\n")
        f.write(f"input_shape={'_'.join(map(str, input_shape))}\n")
        f.write(f"size={'_'.join(map(str, size))}\n")
        f.write(f"stride={'_'.join(map(str, stride))}\n")
        f.write(f"storage_offset={storage_offset}\n")
        f.write(f"input_elements={input_x.size}\n")
        f.write(f"output_elements={output.size}\n")

    return input_x, output


def verify_result(output_npu, output_golden, dtype):
    """Verify NPU output against golden with appropriate tolerance."""
    if dtype == np.int32:
        # Exact match for int32
        match = np.array_equal(output_npu, output_golden)
        max_diff = 0
    else:
        # Tolerance-based comparison for floating point
        if dtype == np.float32:
            rtol, atol = 1e-5, 1e-5
        elif dtype == np.float16:
            rtol, atol = 1e-3, 1e-3
        else:
            rtol, atol = 1e-5, 1e-5

        max_diff = float(np.max(np.abs(output_npu.astype(np.float64) - output_golden.astype(np.float64))))
        match = np.allclose(output_npu, output_golden, rtol=rtol, atol=atol)

    return match, max_diff


if __name__ == "__main__":
    # Generate all test data
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test", "data")
    os.makedirs(output_dir, exist_ok=True)

    for tc in TEST_CASES:
        input_x, output = gen_test_data(tc, output_dir)
        print(f"Generated: {tc['name']} input={input_x.shape} output={output.shape}")

    print(f"\nAll test data generated in {output_dir}")
