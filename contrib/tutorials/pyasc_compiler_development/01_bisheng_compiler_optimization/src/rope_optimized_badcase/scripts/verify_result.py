# ============================================================================
# 结果验证脚本 - rope_optimized
# ============================================================================
#
# 用法: python verify_result.py <output.bin> <golden.bin> [dtype]
#   dtype: float32 或 float16（默认 float32）

import numpy as np
import sys


def verify_result(output_path, golden_path, dtype=np.float32):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    # 精度标准依据 PLAN.md §4.2
    if dtype == np.float16:
        rtol = 1e-3
        atol = 1e-3
    else:
        rtol = 1e-5
        atol = 1e-5

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(output - golden))
        mean_diff = np.mean(np.abs(output - golden))
        print(f"Verification PASSED! Shape: {output.shape}, dtype: {dtype}")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  rtol={rtol}, atol={atol}")
        return True
    else:
        diff = np.abs(output - golden)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"Verification FAILED!")
        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"  rtol={rtol}, atol={atol}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"  Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            print(f"  First 10 mismatch indices: {mismatches[:10]}")
            for idx in mismatches[:5]:
                print(f"    [{idx}] output={output[idx]:.6f}, golden={golden[idx]:.6f}, diff={diff[idx]:.6e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin> [dtype]")
        sys.exit(1)

    dtype = np.float32
    if len(sys.argv) >= 4:
        dt = sys.argv[3].lower()
        if dt in ("half", "fp16", "float16"):
            dtype = np.float16
        elif dt in ("float", "fp32", "float32"):
            dtype = np.float32

    success = verify_result(sys.argv[1], sys.argv[2], dtype)
    sys.exit(0 if success else 1)
