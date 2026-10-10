import tilelang
import tilelang.ascend.language as T


def gemm(M=512, N=36864, K=256, BM=256, BN=128, stages=2, persistent=False):
    assert M % BM == 0 and 36 % (M // BM) == 0
    cores_n = 36 // (M // BM)
    assert N % (cores_n * BN) == 0
    assert 2 * BM * K + stages * 2 * BN * K <= 512 * 1024
    assert stages * 4 * BM * BN <= 256 * 1024

    @T.prim_func
    def main(
        A: T.Tensor((M, K), 'bfloat16'),
        B: T.Tensor((N, K), 'bfloat16'),
        C: T.Tensor((M, N), 'float32'),
    ):
        with T.Kernel(36) as bx:
            a = T.alloc_l1((BM, K), 'bfloat16')
            b = T.alloc_l1((BN, K), 'bfloat16')
            c = T.alloc_l0c((BM, BN), 'float32')
            T.annotate_buffer_versions({a: 1, b: stages, c: stages})
            mi = bx % (M // BM)
            T.copy(A[mi * BM : (mi + 1) * BM, :], a)
            for ni in T.Persistent([N // BN], cores_n, bx // (M // BM), num_stages=stages):
                T.copy(B[ni * BN : (ni + 1) * BN, :], b)
                T.gemm(a, b, c, transpose_B=True, clear_accum=True)

                T.copy(c, C[mi * BM : (mi + 1) * BM, ni * BN : (ni + 1) * BN])

    return main
