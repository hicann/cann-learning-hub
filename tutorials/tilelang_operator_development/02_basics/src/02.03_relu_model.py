"""Chapter 2.3: CopyIn -> Compute -> CopyOut with a single-tile ReLU."""
import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def make_relu(n: int = 1024):
    if n <= 0 or n % 64 != 0:
        raise ValueError(f"n ({n}) must be a positive multiple of 64 for float32 SimdVF")
    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), T.float32)
            y_ub = T.alloc_shared((n,), T.float32)
            T.copy(A[:], x_ub)
            with T.SimdVF():
                # One register holds 64 float32 lanes on this Ascend vector unit.
                mask = S.pset(32, "PAT_ALL")
                zero = S.vdup(T.float32(0), "float32", mask)
                for r in T.serial(n // 64):
                    x_vec = S.vld(x_ub[r * 64])
                    S.vsts(y_ub[r * 64], S.vmax(x_vec, zero, mask), mask)
            T.copy(y_ub, C[:])

    return main


def run(n: int = 1024) -> None:
    if n <= 0 or n % 64 != 0:
        raise ValueError(f"n ({n}) must be a positive multiple of 64 for float32 SimdVF")
    torch.manual_seed(23)
    x = torch.randn(n, device="npu", dtype=torch.float32)
    kernel = tilelang.compile(make_relu(n), target="ascend", out_idx=-1)
    y = kernel(x)
    torch.npu.synchronize()
    expected = torch.relu(x)
    torch.testing.assert_close(y, expected, rtol=0.0, atol=0.0)
    print(f"ReLU shape=({n},), negative_values={(x < 0).sum().item()}")
    print("Verification passed!")


if __name__ == "__main__":
    run()
