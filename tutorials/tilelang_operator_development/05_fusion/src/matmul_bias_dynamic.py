import tilelang
import tilelang.language as T
from tilelang.language import simd as S

tilelang.disable_cache()

N = K = 128
BM = BN = BK = 64
M = T.dynamic("num_rows")  # ① M是运行时行数

@T.prim_func
def matmul_bias_dynamic(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((N, K), "float16"),
    Bias: T.Tensor((N,), "float32"),
    Y: T.Tensor((M, N), "float32"),
):
    T.assume(M > 0)  # ② 调用方保证的条件
    T.assume(M % BM == 0)

    # ③ 任务数根据本次M计算
    with T.Kernel((M // BM) * (N // BN)) as bx:
        row = bx // (N // BN)
        col = bx % (N // BN)

        a = T.alloc_l1((BM, BK), "float16")
        b = T.alloc_l1((BN, BK), "float16")
        acc = T.alloc_l0c((BM, BN), "float32")
        v = T.alloc_shared((BM // 2, BN), "float32")
        bias = T.alloc_shared((BN,), "float32")

        for ki in T.serial(K // BK):
            T.copy(A[row * BM:(row + 1) * BM,
                     ki * BK:(ki + 1) * BK], a)
            T.copy(B[col * BN:(col + 1) * BN,
                     ki * BK:(ki + 1) * BK], b)
            T.gemm(a, b, acc,
                   transpose_B=True, clear_accum=(ki == 0))

        T.dual_copy(acc, v)
        T.copy(Bias[col * BN:(col + 1) * BN], bias)

        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for i in T.serial(BM // 2):
                value = S.vld(v[i, 0])
                bv = S.vld(bias[0])
                result = S.vadd(value, bv, full)
                S.vsts(v[i, 0], result, full)

        T.dual_copy(v, Y[row * BM:(row + 1) * BM,
                         col * BN:(col + 1) * BN])

import torch
import torch_npu

def check_inputs(a, b, bias):
    if a.ndim != 2 or b.shape != (N, K) or bias.shape != (N,):
        raise ValueError("输入形状不符合要求")
    if a.shape[1] != K:
        raise ValueError("A的列数必须等于K")
    m = a.shape[0]
    if m <= 0 or m % BM != 0:
        raise ValueError("M必须为64的正整数倍")
    if a.dtype != torch.float16 or b.dtype != torch.float16 or bias.dtype != torch.float32:
        raise ValueError("A、B必须为float16，Bias必须为float32")
    if not all(x.is_contiguous() for x in (a, b, bias)):
        raise ValueError("输入必须连续存放")
    return m
torch.npu.set_device(0)
kernel = tilelang.compile(matmul_bias_dynamic, target="ascend", out_idx=-1)

b_cpu = torch.ones((N, K), dtype=torch.float16)
bias_cpu = torch.arange(N, dtype=torch.float32)
b_npu, bias_npu = b_cpu.npu(), bias_cpu.npu()

for m in (64, 128, 192):
    a_cpu = torch.ones((m, K), dtype=torch.float16)
    check_inputs(a_cpu, b_cpu, bias_cpu)
    actual = kernel(a_cpu.npu(), b_npu, bias_npu).cpu()
    expected = a_cpu.float() @ b_cpu.float().T + bias_cpu
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    print(f"M={m}，输出形状={tuple(actual.shape)}，完整输出比较通过")
    print("第0行的第0、63、64、127列：", actual[0, [0, 63, 64, 127]].tolist())

print("Verification passed!")
