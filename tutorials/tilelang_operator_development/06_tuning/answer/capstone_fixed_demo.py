import tilelang
import tilelang.language as T
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

def make_inputs(kind):
    torch.manual_seed(20260915)
    if kind == "random":
        a = (torch.randn(M, K) * 0.125).half()
        b = (torch.randn(N, K) * 0.125).half()
    elif kind == "zero":
        a = torch.zeros((M, K), dtype=torch.float16)
        b = torch.zeros((N, K), dtype=torch.float16)
    elif kind == "constant":
        a = torch.full((M, K), 0.125, dtype=torch.float16)
        b = torch.full((N, K), -0.25, dtype=torch.float16)
    else:
        a = ((torch.arange(M)[:, None] % 5 - 2).expand(M, K).float() / 16).half().contiguous()
        b = ((torch.arange(N)[:, None] % 7 - 3).expand(N, K).float() / 16).half().contiguous()
    return a, b

torch.npu.set_device(0)
a_cpu = torch.ones((M, K), dtype=torch.float16)
b_cpu = torch.arange(1, N + 1, dtype=torch.float16)[:, None].expand(N, K).contiguous()
expected = a_cpu.float() @ b_cpu.float().T
fixed = tilelang.compile(make_gemm(stages=1, shift_b=False),
                         target="ascend", out_idx=-1)
fixed_actual = fixed(a_cpu.npu(), b_cpu.npu()).cpu()
torch.testing.assert_close(fixed_actual, expected, rtol=0.002, atol=0.002)
print("修复后：C[0,0]、C[0,64] =", fixed_actual[0, [0, 64]].tolist())
fixed_checks = []
for kind in ("random", "zero", "constant", "indexed"):
    a_cpu, b_cpu = make_inputs(kind)
    actual = fixed(a_cpu.npu(), b_cpu.npu()).cpu()
    expected = a_cpu.float() @ b_cpu.float().T
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    fixed_checks.append({"case": kind,
                         "max_abs_error": (actual - expected).abs().max().item()})
    print(f"修复后：{kind}，全部{M * N}个元素比较通过")
print("Verification passed!")
