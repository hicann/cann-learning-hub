import os
os.environ["TILELANG_DISABLE_CACHE"] = "1"

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
import torch
import torch_npu

N = 64

def make_kernel():
    @T.prim_func
    def demo_vdup(Y: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            y_ub = T.alloc_shared((N,), "float32")
            with T.SimdVF():
                full = S.pset(32)
                two = S.vdup(T.float32(2), "float32", full)
                S.vsts(y_ub[0], two, full)
            T.copy(y_ub, Y)
    return demo_vdup

def main():
    torch.npu.set_device(0)
    kernel = tilelang.compile(make_kernel(), target="ascend", out_idx=-1)
    output = kernel().cpu()
    expected = torch.full((N,), 2.0, dtype=torch.float32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print("output[:16] =", output[:16].tolist())
    print("all 64 elements passed")
    print("Verification passed!")


if __name__ == "__main__":
    main()
