import tilelang
import tilelang.language as T
from tilelang.language import simd as S

tilelang.disable_cache()

M = N = K = 128
BM = BN = BK = 64

@T.prim_func
def matmul_bias(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((N, K), "float16"),
    Bias: T.Tensor((N,), "float32"),
    Y: T.Tensor((M, N), "float32"),
):
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

torch.npu.set_device(0)
a_cpu = torch.ones((M, K), dtype=torch.float16)
b_cpu = torch.ones((N, K), dtype=torch.float16)
bias_cpu = torch.arange(N, dtype=torch.float32)

kernel = tilelang.compile(matmul_bias, target="ascend", out_idx=-1)
actual = kernel(a_cpu.npu(), b_cpu.npu(), bias_cpu.npu()).cpu()
expected = a_cpu.float() @ b_cpu.float().T + bias_cpu
torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)

print(actual[0, [0, 63, 64, 127]].tolist())
print("Verification passed!")
