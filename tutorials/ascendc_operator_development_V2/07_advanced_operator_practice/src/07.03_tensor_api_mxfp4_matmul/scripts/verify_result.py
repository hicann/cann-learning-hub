import sys

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16

RELATIVE_TOL = 1e-3
ABSOLUTE_TOL = 1e-3
ERROR_TOL = 1e-3


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 verify_result.py output/<case>.bin")

    output_path = sys.argv[1]
    golden_path = "output/golden.bin"

    output = np.fromfile(output_path, dtype=bfloat16).reshape(-1)
    golden = np.fromfile(golden_path, dtype=bfloat16).reshape(-1)

    if output.size != golden.size:
        raise SystemExit(f"size mismatch: output {output.size}, golden {golden.size}")

    different = ~np.isclose(
        output.astype(np.float32),
        golden.astype(np.float32),
        rtol=RELATIVE_TOL,
        atol=ABSOLUTE_TOL,
        equal_nan=True,
    )
    diff_idx = np.where(different)[0]
    error_count = diff_idx.size
    total = golden.size

    print(f"Total elements compared: {total}")
    print(f"Total error elements: {error_count}")

    if error_count > 0:
        for idx in diff_idx[:100]:
            golden_val = float(golden[idx])
            output_val = float(output[idx])
            denom = abs(golden_val) if abs(golden_val) > 1e-12 else 1.0
            print(f"data index: {idx:06d}, expected: {golden_val:.9f}, actual: {output_val:.9f}, "
                  f"rdiff: {abs(output_val - golden_val) / denom:.6f}")

    error_ratio = float(error_count) / total
    print(f"error ratio: {error_ratio:.4f}, tolerance: {ERROR_TOL:.4f}")

    if error_ratio > ERROR_TOL:
        raise SystemExit("verify failed!")
    print("test pass!")


if __name__ == "__main__":
    main()
