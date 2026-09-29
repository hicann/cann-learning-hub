import os
os.environ["TILELANG_DISABLE_CACHE"] = "1"

import tilelang
import tilelang.language as T
from tilelang.language import simd as S
import torch
import torch_npu

N = 64

def make_kernel():
    @T.prim_func
    def demo_vcvt(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float8_e4m3fn")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), "float32")
            y_ub = T.alloc_shared((N,), "float8_e4m3fn")
            T.copy(X, x_ub)
            with T.SimdVF():
                x = S.vld(x_ub[0])
                q = S.vcvt(x, "float8_e4m3fn", round="ROUND_R", sat=True, part=0)
                S.vsts(y_ub[0], q, dist="PK4_B32")
            T.copy(y_ub, Y)
    return demo_vcvt


def check_input(x_cpu):
    if x_cpu.shape != (N,) or x_cpu.dtype != torch.float32 or not x_cpu.is_contiguous():
        raise ValueError("expected 64 contiguous float32 elements")
    if not torch.isfinite(x_cpu).all() or x_cpu.abs().max() > 448:
        raise ValueError("this experiment requires finite inputs in [-448, 448]")


def main():
    torch.npu.set_device(0)
    kernel = tilelang.compile(make_kernel(), target="ascend", out_idx=-1)
    generator = torch.Generator().manual_seed(20260916)
    values = [-400, -448, -1.1875, -1.0625, -0.1, -0.0, 0.0, 0.1,
              1, 1.0625, 1.1875, 1.25, 240, 448, 400, 0.001953125]
    cases = {
        "mapping": torch.tensor(values, dtype=torch.float32).repeat(4),
        "zero": torch.zeros(N, dtype=torch.float32),
        "constant": torch.full((N,), 1.0625, dtype=torch.float32),
        "mixed": torch.randint(-1792, 1793, (N,), generator=generator).float() / 4,
    }
    for name, x_cpu in cases.items():
        check_input(x_cpu)
        q_cpu = kernel(x_cpu.npu()).cpu()
        expected = x_cpu.to(torch.float8_e4m3fn)
        torch.testing.assert_close(q_cpu.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        if name == "mapping":
            print("input[:16] =", x_cpu[:16].tolist())
            print("fp8 values[:16] =", q_cpu.float()[:16].tolist())
            print("fp8 bytes[:16] =", q_cpu.view(torch.uint8)[:16].tolist())
        print(f"{name}: all {N} FP8 encodings passed")
    for value in (-500.0, 500.0, float("inf"), float("nan")):
        try:
            check_input(torch.full((N,), value, dtype=torch.float32))
        except ValueError as error:
            print(f"invalid value {value} rejected:", error)
        else:
            raise AssertionError("unsupported input was accepted")
    print("Verification passed!")


if __name__ == "__main__":
    main()
