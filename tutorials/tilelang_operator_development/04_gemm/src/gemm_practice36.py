import tilelang
import tilelang.ascend.language as T

BLOCKS = 36


def check_config(M, N, K, BM, BN, stages):
    if any(type(x) is not int or x <= 0 for x in (M, N, K, BM, BN, stages)):
        raise ValueError('Dimensions and stages must be positive integers')
    if BM not in (64, 128, 256) or BN not in (64, 128, 256) or K not in (64, 128, 256):
        raise ValueError('Teaching domain: BM/BN=64/128/256; full K=64/128/256')
    if stages not in (1, 2, 3) or M % BM or N % BN:
        raise ValueError('Stages=1/2/3; output element tails are not implemented')
    if (M // BM) * (N // BN) % BLOCKS:
        raise ValueError('This comparison uses equal complete tasks on 36 cores')
    if stages * 2 * (BM + BN) * K > 512 * 1024:
        raise ValueError('L1 buffer versions exceed the 512 KiB budget')
    if stages * 4 * BM * BN > 256 * 1024:
        raise ValueError('L0C buffer versions exceed the 256 KiB budget')


def gemm(M=512, N=36864, K=256, BM=128, BN=128, stages=1, persistent=False):
    check_config(M, N, K, BM, BN, stages)
    tasks = (M // BM) * (N // BN)
    blocks = BLOCKS
    s = stages
    mode = 'persistent' if persistent else 'contiguous'

    @T.prim_func
    def main(
        A: T.Tensor((M, K), 'bfloat16'),
        B: T.Tensor((N, K), 'bfloat16'),
        C: T.Tensor((M, N), 'float32'),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((BM, K), 'bfloat16')
            b = T.alloc_l1((BN, K), 'bfloat16')
            c = T.alloc_l0c((BM, BN), 'float32')
            T.annotate_buffer_versions({a: s, b: s, c: s})
            if mode == 'persistent':
                for idx in T.Persistent([tasks], blocks, bx, num_stages=s):
                    mi = idx // (N // BN)
                    ni = idx % (N // BN)
                    T.copy(A[mi * BM : (mi + 1) * BM, :], a)
                    T.copy(B[ni * BN : (ni + 1) * BN, :], b)
                    T.gemm(a, b, c, transpose_B=True, clear_accum=True)

                    T.copy(c, C[mi * BM : (mi + 1) * BM, ni * BN : (ni + 1) * BN])
            else:
                for it in T.Pipelined(tasks // blocks, num_stages=s):
                    idx = bx * (tasks // blocks) + it
                    mi = idx // (N // BN)
                    ni = idx % (N // BN)
                    T.copy(A[mi * BM : (mi + 1) * BM, :], a)
                    T.copy(B[ni * BN : (ni + 1) * BN, :], b)
                    T.gemm(a, b, c, transpose_B=True, clear_accum=True)

                    T.copy(c, C[mi * BM : (mi + 1) * BM, ni * BN : (ni + 1) * BN])

    return main
