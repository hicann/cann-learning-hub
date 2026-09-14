import sys

import numpy as np

# 量化误差容忍：元素间允许 ±1 的舍入差异
ERROR_TOL = 0.01  # 1% 的元素允许相差超过 1


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 verify_result_quant.py output/<case>.bin")

    output_path = sys.argv[1]
    golden_path = "output/golden_quant.bin"

    output = np.fromfile(output_path, dtype=np.int8).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.int8).reshape(-1)

    if output.size != golden.size:
        raise SystemExit(f"size mismatch: output {output.size}, golden {golden.size}")

    diff = np.abs(output.astype(np.int32) - golden.astype(np.int32))
    # 误差超过 1 的元素视为错误（量化舍入允许 ±1）
    error_mask = diff > 1
    error_count = np.count_nonzero(error_mask)
    total = golden.size

    print(f"Total elements compared: {total}")
    print(f"Total error elements (diff > 1): {error_count}")

    if error_count > 0:
        error_idx = np.where(error_mask)[0]
        for idx in error_idx[:100]:
            golden_val = int(golden[idx])
            output_val = int(output[idx])
            print(f"data index: {idx:06d}, expected: {golden_val}, actual: {output_val}, "
                  f"diff: {abs(output_val - golden_val)}")

    error_ratio = float(error_count) / total
    print(f"error ratio: {error_ratio:.4f}, tolerance: {ERROR_TOL:.4f}")

    if error_ratio > ERROR_TOL:
        raise SystemExit("verify failed!")
    print("test pass!")


if __name__ == "__main__":
    main()
