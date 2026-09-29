import tilelang
import tilelang.language as T


def k_demo_gemm(M=512, N=36864, K=2048, BM=128, BN=128, BK=128):
    if min(M, N, K, BM, BN, BK) <= 0 or M % BM or N % BN or K % BK:
        raise ValueError('Positive dimensions and complete tiles are required')
    tasks = (M // BM) * (N // BN)
    if tasks % 8 or 2 * (BM + BN) * BK > 512 * 1024 or 4 * BM * BN > 256 * 1024:
        raise ValueError('Check task division and L1/L0C capacity')

    @T.prim_func
    def main(
        A: T.Tensor((M, K), 'bfloat16'),
        B: T.Tensor((N, K), 'bfloat16'),
        C: T.Tensor((M, N), 'float32'),
    ):
        with T.Kernel(8) as bx:
            a = T.alloc_l1((BM, BK), 'bfloat16')
            b = T.alloc_l1((BN, BK), 'bfloat16')
            c = T.alloc_l0c((BM, BN), 'float32')

            for it in T.Pipelined(tasks // 8, num_stages=1):
                idx = bx * (tasks // 8) + it
                mi = idx // (N // BN)
                ni = idx % (N // BN)
                for kt in T.Pipelined(K // BK, num_stages=1):
                    T.copy(A[mi * BM : (mi + 1) * BM, kt * BK : (kt + 1) * BK], a)
                    T.copy(B[ni * BN : (ni + 1) * BN, kt * BK : (kt + 1) * BK], b)
                    T.gemm(a, b, c, transpose_B=True, clear_accum=(kt == 0))

                T.copy(c, C[mi * BM : (mi + 1) * BM, ni * BN : (ni + 1) * BN])

    return main
