import sys

import numpy as np

RELATIVE_TOL = 1e-3
ABSOLUTE_TOL = 1e-3
ERROR_TOL = 1e-3


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 verify_result.py output/<case>.bin")

    output_path = sys.argv[1]
    golden_path = "output/golden.bin"

    output = np.fromfile(output_path, dtype=np.float16).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.float16).reshape(-1)

    if output.size != golden.size:
        raise SystemExit(f"size mismatch: output {output.size}, golden {golden.size}")

    close_mask = np.isclose(output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL, equal_nan=True)
    error_indexes = np.where(close_mask == False)[0]

    for idx in error_indexes[:100]:
        golden_val = float(golden[idx])
        output_val = float(output[idx])
        rdiff = abs(output_val - golden_val) / abs(golden_val) if golden_val != 0 else abs(output_val - golden_val)
        print(f"data index: {idx:06d}, expected: {golden_val:.9f}, actual: {output_val:.9f}, rdiff: {rdiff:.6f}")

    error_ratio = float(error_indexes.size) / golden.size
    print(f"error ratio: {error_ratio:.4f}, tolerance: {ERROR_TOL:.4f}")

    if error_ratio > ERROR_TOL:
        raise SystemExit("verify failed!")
    print("test pass!")


if __name__ == "__main__":
    main()
