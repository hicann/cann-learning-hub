import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

tilelang.disable_cache()  # 本例每次运行都重新编译。

N = 1024
TILE = 512


# 1. 描述设备上怎样计算。
@T.prim_func
def vector_add(
    A: T.Buffer((N,), "float32"),
    B: T.Buffer((N,), "float32"),
    C: T.Buffer((N,), "float32"),
):
    with T.Kernel(N // TILE) as bx:
        a_ub = T.alloc_shared((TILE,), "float32")
        b_ub = T.alloc_shared((TILE,), "float32")
        c_ub = T.alloc_shared((TILE,), "float32")

        # 搬入当前数据块。
        begin = bx * TILE
        T.copy(A[begin : begin + TILE], a_ub)
        T.copy(B[begin : begin + TILE], b_ub)

        # 每轮加载一组数据，做向量加法，再存回UB。
        with T.SimdVF():
            mask = T.simd.pset(32)
            for r in range(TILE // 64):
                a_vec = T.simd.vld(a_ub[r * 64])
                b_vec = T.simd.vld(b_ub[r * 64])
                c_vec = T.simd.vadd(a_vec, b_vec, mask)
                T.simd.vsts(c_ub[r * 64], c_vec, mask)

        # 写回这一块结果。
        T.copy(c_ub, C[begin : begin + TILE])


# 2. 编译计算描述，得到可以调用的kernel。
kernel = tilelang.compile(vector_add, target="ascend", out_idx=-1)

# 3. 准备输入，调用kernel，再核对结果。
a = torch.arange(N, device="npu", dtype=torch.float32)
b = torch.ones(N, device="npu", dtype=torch.float32)
out = kernel(a, b)
torch.testing.assert_close(out, a + b, rtol=0, atol=0)
print("out[:8] =", out[:8].cpu().tolist())
print("Verification passed! (已比较全部1024项输出)")
