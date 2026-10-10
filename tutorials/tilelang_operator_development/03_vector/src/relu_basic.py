"""One logical task: whole-input ReLU/Exp, finite contiguous fp32.
The computation uses explicit SIMD register instructions inside SimdVF.
"""

import tilelang
import tilelang.ascend.language as T
from vector_kernels import check_config


def make_relu(N):
    check_config(N, N, 1, 1, 2)

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(1) as bx:
            a = T.alloc_shared((N,), 'float32')
            c = T.alloc_shared((N,), 'float32')
            T.copy(A, a)

            with T.SimdVF():
                mask = T.simd.pset(32)
                zero = T.simd.vdup(0.0, "float32", mask)
                for group in T.serial(N // 64):
                    ra = T.simd.vld(a[group * 64])
                    rc = T.simd.vmax(ra, zero, mask)
                    T.simd.vsts(c[group * 64], rc, mask)

            T.copy(c, C)

    return tilelang.compile(main, out_idx=-1)


def make_exp(N):
    check_config(N, N, 1, 1, 2)

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(1) as bx:
            a = T.alloc_shared((N,), 'float32')
            c = T.alloc_shared((N,), 'float32')
            T.copy(A, a)

            with T.SimdVF():
                mask = T.simd.pset(32)
                for group in T.serial(N // 64):
                    ra = T.simd.vld(a[group * 64])
                    rc = T.simd.vexp(ra, mask)
                    T.simd.vsts(c[group * 64], rc, mask)

            T.copy(c, C)

    return tilelang.compile(main, out_idx=-1)
