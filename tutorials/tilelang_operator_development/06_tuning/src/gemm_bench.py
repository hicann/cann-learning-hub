import tilelang
import tilelang.language as T
import torch
import torch_npu
from tilelang.profiler import do_bench

tilelang.disable_cache()

M = N = K = 512
BM = BN = BK = 64
WARMUP = 30
REPEAT = 100


@T.prim_func
def lesson_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((N, K), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel((M // BM) * (N // BN)) as bx:
        row = bx // (N // BN)
        col = bx % (N // BN)
        a = T.alloc_l1((BM, BK), "float16")
        b = T.alloc_l1((BN, BK), "float16")
        acc = T.alloc_l0c((BM, BN), "float32")
        for ki in T.Pipelined(K // BK, num_stages=2):
            T.copy(A[row * BM:(row + 1) * BM,
                     ki * BK:(ki + 1) * BK], a)
            T.copy(B[col * BN:(col + 1) * BN,
                     ki * BK:(ki + 1) * BK], b)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=(ki == 0))
        T.copy(acc, C[row * BM:(row + 1) * BM,
                      col * BN:(col + 1) * BN])


def main():
    torch.npu.set_device(0)
    kernel = tilelang.compile(lesson_gemm, target="ascend", out_idx=-1)
    torch.manual_seed(20260916)
    a_cpu = (torch.randn(M, K) * 0.125).half()
    b_cpu = (torch.randn(N, K) * 0.125).half()
    a_npu, b_npu = a_cpu.npu(), b_cpu.npu()

    actual = kernel(a_npu, b_npu).cpu()
    expected = a_cpu.float() @ b_cpu.float().T
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    print("Verification passed!", flush=True)

    latency_ms = do_bench(
        lambda: kernel(a_npu, b_npu),
        backend="msprof",
        _n_warmup=WARMUP,
        _n_repeat=REPEAT,
        cache_size=256,
    )
    print(f"latency_ms = {latency_ms:.8f}")
    print(f"latency_us = {latency_ms * 1000:.5f}")


if __name__ == "__main__":
    main()
