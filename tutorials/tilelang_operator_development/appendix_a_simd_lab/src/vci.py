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
    def demo_vci(Y: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            y_ub = T.alloc_shared((N,), "float32")
            with T.SimdVF():
                full = S.pset(32)
                ramp = S.vci(T.float32(10), "float32", order="INC_ORDER")
                S.vsts(y_ub[0], ramp, full)
            T.copy(y_ub, Y)
    return demo_vci

def main():
    torch.npu.set_device(0)
    kernel = tilelang.compile(make_kernel(), target="ascend", out_idx=-1)
    output = kernel().cpu()
    expected = torch.arange(10, 74, dtype=torch.float32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print("output[:16] =", output[:16].tolist())
    print("all 64 elements passed")
    print("Verification passed!")


if __name__ == "__main__":
    main()
