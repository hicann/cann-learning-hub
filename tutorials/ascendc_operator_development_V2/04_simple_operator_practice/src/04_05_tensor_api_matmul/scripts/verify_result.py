import sys

import numpy as np


M = 8192
N = 8192
K = 8192
ATOL = 1e-3
RTOL = 1e-3


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 verify_result.py output/<case>.bin")

    output_path = sys.argv[1]
    golden_path = "output/golden.bin"

    output = np.fromfile(output_path, dtype=np.float16)
    golden = np.fromfile(golden_path, dtype=np.float16)

    expected_size = M * N
    if output.size != expected_size:
        raise SystemExit(f"invalid output size: {output.size}, expected {expected_size}")
    if golden.size != expected_size:
        raise SystemExit(f"invalid golden size: {golden.size}, expected {expected_size}")

    failed = ~np.isclose(output, golden, atol=ATOL, rtol=RTOL)
    if np.any(failed):
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        diff_idx = np.flatnonzero(failed)
        first = int(diff_idx[0])
        raise SystemExit(
            f"verify failed at flat index {first}: "
            f"got {output[first]}, golden {golden[first]}, abs diff {diff[first]}"
        )

    print(f"verify passed: {output_path}")


if __name__ == "__main__":
    main()
