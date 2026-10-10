"""2.3 ReLU 练习参考实现。"""

import torch
import tilelang
import tilelang.ascend.language as T


@tilelang.jit(out_idx=-1, target="ascend")
def make_relu(n: int = 1024):
    # 当前例子只处理完整的 64 元素组。
    if n <= 0 or n % 64:
        raise ValueError("n 必须是 64 的正整数倍")

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), T.float32)
            y_ub = T.alloc_shared((n,), T.float32)

            # CopyIn：输入从 GM 搬到 UB。
            T.copy(A[:], x_ub)

            # Compute：每次从 UB 读取 64 个元素，在寄存器中计算。
            with T.SimdVF():
                mask = T.simd.pset(32)
                zero = T.simd.vdup(T.float32(0), "float32", mask)
                for k in T.serial(n // 64):
                    x_vec = T.simd.vld(x_ub[k * 64])
                    y_vec = T.simd.vmax(x_vec, zero, mask)
                    T.simd.vsts(y_ub[k * 64], y_vec, mask)

            # CopyOut：结果从 UB 写回 GM。
            T.copy(y_ub, C[:])

    return main


def run():
    n = 1024
    torch.manual_seed(23)
    x = torch.randn(n, device="npu", dtype=torch.float32)

    kernel = make_relu(n)
    actual = kernel(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, torch.relu(x), rtol=0.0, atol=0.0)
    print(f"ReLU：{n} 个元素全部通过比较")


if __name__ == "__main__":
    run()
