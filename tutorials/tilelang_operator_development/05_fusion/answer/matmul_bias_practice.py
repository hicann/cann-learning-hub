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

def make_inputs(m, kind):
    if kind == "random":
        a = (torch.randn(m, K) * 0.125).half()
        b = (torch.randn(N, K) * 0.125).half()
        bias = torch.randn(N)
    elif kind == "zero":
        a = torch.zeros((m, K), dtype=torch.float16)
        b = torch.zeros((N, K), dtype=torch.float16)
        bias = torch.zeros(N)
    elif kind == "constant":
        a = torch.full((m, K), 0.125, dtype=torch.float16)
        b = torch.full((N, K), -0.25, dtype=torch.float16)
        bias = torch.full((N,), 0.5)
    elif kind == "indexed":
        a = ((torch.arange(m)[:, None] % 7 - 3).expand(m, K).float() / 16).half().contiguous()
        b = ((torch.arange(N)[:, None] % 5 - 2).expand(N, K).float() / 8).half().contiguous()
        bias = torch.arange(N).float() / 16
    elif kind == "signed":
        a = (torch.randn(m, K) * 2).half()
        b = (torch.randn(N, K) * 2).half()
        bias = torch.linspace(-4, 4, N)
    else:
        raise ValueError("未知的测试类型")
    return a, b, bias

torch.npu.set_device(0)
torch.manual_seed(20260915)
kernel = tilelang.compile(matmul_bias_dynamic, target="ascend", out_idx=-1)

results = []
for m in (64, 128, 192, 256, 640):
    for kind in ("random", "zero", "constant", "indexed", "signed"):
        a_cpu, b_cpu, bias_cpu = make_inputs(m, kind)
        check_inputs(a_cpu, b_cpu, bias_cpu)
        actual = kernel(a_cpu.npu(), b_cpu.npu(), bias_cpu.npu()).cpu()
        expected = a_cpu.float() @ b_cpu.float().T + bias_cpu
        torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        max_error = (actual - expected).abs().max().item()
        results.append({"M": m, "case": kind, "elements": actual.numel(),
                        "max_abs_error": max_error, "passed": True})
        if m == 64 and kind == "indexed":
            points = [(0, 0), (31, 63), (32, 64), (63, 127)]
            print("按位置构造的输入，四个检查点：",
                  [actual[r, c].item() for r, c in points])
    print(f"M={m}：5类输入的完整输出比较均通过")

rejected = []
for m in (0, 63, 65, 193, 257):
    a_cpu = torch.ones((m, K), dtype=torch.float16)
    b_cpu = torch.ones((N, K), dtype=torch.float16)
    bias_cpu = torch.zeros(N)
    try:
        check_inputs(a_cpu, b_cpu, bias_cpu)
    except ValueError as error:
        rejected.append({"M": m, "error": str(error)})
        print(f"M={m}：调用前拒绝，{error}")
    else:
        raise AssertionError(f"M={m}应当被拒绝")

print(f"Verification passed! {len(results)}组完整输出比较通过，"
      f"{len(rejected)}种不符合要求的行数均在调用前拒绝。")
