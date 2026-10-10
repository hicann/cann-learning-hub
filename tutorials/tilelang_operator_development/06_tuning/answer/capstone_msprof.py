import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu

tilelang.disable_cache()

M = N = K = 128
BM = BN = BK = 64

def make_gemm(stages=1, shift_b=False):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel((M // BM) * (N // BN)) as bx:
            row = bx // (N // BN)
            col = bx % (N // BN)
            b_col = (col + 1) % (N // BN) if shift_b else col
            a = T.alloc_l1((BM, BK), "float16")
            b = T.alloc_l1((BN, BK), "float16")
            acc = T.alloc_l0c((BM, BN), "float32")
            for ki in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[row * BM:(row + 1) * BM,
                         ki * BK:(ki + 1) * BK], a)
                T.copy(B[b_col * BN:(b_col + 1) * BN,
                         ki * BK:(ki + 1) * BK], b)
                T.gemm(a, b, acc,
                       transpose_B=True, clear_accum=(ki == 0))
            T.copy(acc, C[row * BM:(row + 1) * BM,
                          col * BN:(col + 1) * BN])
    return gemm

torch.npu.set_device(0)
fixed = tilelang.compile(make_gemm(stages=1, shift_b=False),
                         target="ascend", out_idx=-1)
a_cpu = torch.ones((M, K), dtype=torch.float16)
b_cpu = torch.arange(1, N + 1, dtype=torch.float16)[:, None].expand(N, K).contiguous()
a_npu, b_npu = a_cpu.npu(), b_cpu.npu()

c_npu = fixed(a_npu, b_npu)
actual = c_npu.cpu()
expected = a_cpu.float() @ b_cpu.float().T
torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
print("C[0,0], C[0,64] =", actual[0, [0, 64]].tolist())
print("Verification passed!", flush=True)
