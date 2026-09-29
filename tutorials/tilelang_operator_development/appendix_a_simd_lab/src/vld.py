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
    def demo_vld(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), "float32")
            y_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF():
                full = S.pset(32)
                x = S.vld(x_ub[0])
                S.vsts(y_ub[0], x, full)
            T.copy(y_ub, Y)
    return demo_vld

def main():
    torch.npu.set_device(0)
    kernel = tilelang.compile(make_kernel(), target="ascend", out_idx=-1)
    generator = torch.Generator().manual_seed(20260916)
    cases = {
        "indexed": torch.arange(N, dtype=torch.float32) - 3,
        "zero": torch.zeros(N, dtype=torch.float32),
        "constant": torch.full((N,), -2.0, dtype=torch.float32),
        "mixed": torch.randint(-32, 33, (N,), generator=generator).float() / 8,
    }
    for name, x_cpu in cases.items():
        output = kernel(x_cpu.npu()).cpu()
        expected = x_cpu
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if name == "indexed":
            print("input[:8] =", x_cpu[:8].tolist())
            print("output[:8] =", output[:8].tolist())
        print(f"{name}: complete output passed")
    print("Verification passed!")


if __name__ == "__main__":
    main()
