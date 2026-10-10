"""Stable row Softmax: single core, tiled multicore, then pipeline.
Five outputs are retained consistently for stepwise teaching and fair comparisons.
Register-instruction implementation; validation status is recorded separately.
"""

import tilelang
import tilelang.ascend.language as T


def check_softmax_config(M, C, rows, blocks, versions=1):
    if min(M, C, rows, blocks, versions) <= 0 or C % 64 or rows % 8 or M % (rows * blocks):
        raise ValueError("Require C % 64 == 0, rows % 8 == 0, M % (rows * blocks) == 0")
    required = 4 * (4 * rows * C + 2 * rows) * versions
    if required > 248 * 1024:
        raise ValueError("Six fp32 buffers exceed the 248 KiB SimdVF UB budget")


def softmax_single(M=8, C=128):
    rows, blocks = M, 1
    check_softmax_config(M, C, rows, blocks)

    @T.prim_func
    def main(
        A: T.Tensor((M, C), "float32"),
        Max: T.Tensor((M,), "float32"),
        Shift: T.Tensor((M, C), "float32"),
        Exp: T.Tensor((M, C), "float32"),
        Sum: T.Tensor((M,), "float32"),
        Y: T.Tensor((M, C), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((rows, C), 'float32')
            mx = T.alloc_shared((rows,), 'float32')
            shifted = T.alloc_shared((rows, C), 'float32')
            e = T.alloc_shared((rows, C), 'float32')
            sm = T.alloc_shared((rows,), 'float32')
            y = T.alloc_shared((rows, C), 'float32')
            T.copy(A, a)

            with T.SimdVF():
                mask = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), "float32")
                for i in T.serial(rows):
                    acc[0] = T.simd.vld(a[i, 0])
                    for group in T.serial(1, C // 64):
                        ra = T.simd.vld(a[i, group * 64])
                        acc[0] = T.simd.vmax(acc[0], ra, mask)
                    row_max = T.simd.vcmax(acc[0], mask)
                    T.simd.vsts(mx[i], row_max, one, "ONEPT_B32")

            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in T.serial(rows):
                    rmax = T.simd.vld(mx[i], "BRC_B32")
                    for group in T.serial(C // 64):
                        ra = T.simd.vld(a[i, group * 64])
                        rs = T.simd.vsub(ra, rmax, mask)
                        T.simd.vsts(shifted[i, group * 64], rs, mask)

            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in T.serial(rows):
                    for group in T.serial(C // 64):
                        rs = T.simd.vld(shifted[i, group * 64])
                        re = T.simd.vexp(rs, mask)
                        T.simd.vsts(e[i, group * 64], re, mask)

            with T.SimdVF():
                mask = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), "float32")
                for i in T.serial(rows):
                    acc[0] = T.simd.vld(e[i, 0])
                    for group in T.serial(1, C // 64):
                        re = T.simd.vld(e[i, group * 64])
                        acc[0] = T.simd.vadd(acc[0], re, mask)
                    row_sum = T.simd.vcadd(acc[0], mask)
                    T.simd.vsts(sm[i], row_sum, one, "ONEPT_B32")

            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in T.serial(rows):
                    rsum = T.simd.vld(sm[i], "BRC_B32")
                    for group in T.serial(C // 64):
                        re = T.simd.vld(e[i, group * 64])
                        ry = T.simd.vdiv(re, rsum, mask)
                        T.simd.vsts(y[i, group * 64], ry, mask)

            T.copy(mx, Max)
            T.copy(shifted, Shift)
            T.copy(e, Exp)
            T.copy(sm, Sum)
            T.copy(y, Y)

    return tilelang.compile(main, out_idx=[1, 2, 3, 4, 5])


def softmax_kernel(M, C, rows=8, blocks=1):
    check_softmax_config(M, C, rows, blocks)

    @T.prim_func
    def main(
        A: T.Tensor((M, C), "float32"),
        Max: T.Tensor((M,), "float32"),
        Shift: T.Tensor((M, C), "float32"),
        Exp: T.Tensor((M, C), "float32"),
        Sum: T.Tensor((M,), "float32"),
        Y: T.Tensor((M, C), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((rows, C), 'float32')
            mx = T.alloc_shared((rows,), 'float32')
            shifted = T.alloc_shared((rows, C), 'float32')
            e = T.alloc_shared((rows, C), 'float32')
            sm = T.alloc_shared((rows,), 'float32')
            y = T.alloc_shared((rows, C), 'float32')

            for it in T.serial(M // (rows * blocks)):
                begin = bx * (M // blocks) + it * rows
                T.copy(A[begin : begin + rows, 0:C], a)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    acc = T.simd.alloc_local((1,), "float32")
                    for i in T.serial(rows):
                        acc[0] = T.simd.vld(a[i, 0])
                        for group in T.serial(1, C // 64):
                            ra = T.simd.vld(a[i, group * 64])
                            acc[0] = T.simd.vmax(acc[0], ra, mask)
                        row_max = T.simd.vcmax(acc[0], mask)
                        T.simd.vsts(mx[i], row_max, one, "ONEPT_B32")

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        rmax = T.simd.vld(mx[i], "BRC_B32")
                        for group in T.serial(C // 64):
                            ra = T.simd.vld(a[i, group * 64])
                            rs = T.simd.vsub(ra, rmax, mask)
                            T.simd.vsts(shifted[i, group * 64], rs, mask)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        for group in T.serial(C // 64):
                            rs = T.simd.vld(shifted[i, group * 64])
                            re = T.simd.vexp(rs, mask)
                            T.simd.vsts(e[i, group * 64], re, mask)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    acc = T.simd.alloc_local((1,), "float32")
                    for i in T.serial(rows):
                        acc[0] = T.simd.vld(e[i, 0])
                        for group in T.serial(1, C // 64):
                            re = T.simd.vld(e[i, group * 64])
                            acc[0] = T.simd.vadd(acc[0], re, mask)
                        row_sum = T.simd.vcadd(acc[0], mask)
                        T.simd.vsts(sm[i], row_sum, one, "ONEPT_B32")

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        rsum = T.simd.vld(sm[i], "BRC_B32")
                        for group in T.serial(C // 64):
                            re = T.simd.vld(e[i, group * 64])
                            ry = T.simd.vdiv(re, rsum, mask)
                            T.simd.vsts(y[i, group * 64], ry, mask)

                T.copy(mx, Max[begin : begin + rows])
                T.copy(shifted, Shift[begin : begin + rows, 0:C])
                T.copy(e, Exp[begin : begin + rows, 0:C])
                T.copy(sm, Sum[begin : begin + rows])
                T.copy(y, Y[begin : begin + rows, 0:C])

    return tilelang.compile(main, out_idx=[1, 2, 3, 4, 5])


def softmax_pipeline(M, C, rows=8, blocks=36, stages=2):
    check_softmax_config(M, C, rows, blocks, stages)

    @T.prim_func
    def main(
        A: T.Tensor((M, C), "float32"),
        Max: T.Tensor((M,), "float32"),
        Shift: T.Tensor((M, C), "float32"),
        Exp: T.Tensor((M, C), "float32"),
        Sum: T.Tensor((M,), "float32"),
        Y: T.Tensor((M, C), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((rows, C), 'float32')
            mx = T.alloc_shared((rows,), 'float32')
            shifted = T.alloc_shared((rows, C), 'float32')
            e = T.alloc_shared((rows, C), 'float32')
            sm = T.alloc_shared((rows,), 'float32')
            y = T.alloc_shared((rows, C), 'float32')
            T.annotate_buffer_versions({a: stages, mx: stages, shifted: stages, e: stages, sm: stages, y: stages})

            for it in T.Pipelined(M // (rows * blocks), num_stages=stages):
                begin = bx * (M // blocks) + it * rows
                T.copy(A[begin : begin + rows, 0:C], a)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    acc = T.simd.alloc_local((1,), "float32")
                    for i in T.serial(rows):
                        acc[0] = T.simd.vld(a[i, 0])
                        for group in T.serial(1, C // 64):
                            ra = T.simd.vld(a[i, group * 64])
                            acc[0] = T.simd.vmax(acc[0], ra, mask)
                        row_max = T.simd.vcmax(acc[0], mask)
                        T.simd.vsts(mx[i], row_max, one, "ONEPT_B32")

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        rmax = T.simd.vld(mx[i], "BRC_B32")
                        for group in T.serial(C // 64):
                            ra = T.simd.vld(a[i, group * 64])
                            rs = T.simd.vsub(ra, rmax, mask)
                            T.simd.vsts(shifted[i, group * 64], rs, mask)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        for group in T.serial(C // 64):
                            rs = T.simd.vld(shifted[i, group * 64])
                            re = T.simd.vexp(rs, mask)
                            T.simd.vsts(e[i, group * 64], re, mask)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    acc = T.simd.alloc_local((1,), "float32")
                    for i in T.serial(rows):
                        acc[0] = T.simd.vld(e[i, 0])
                        for group in T.serial(1, C // 64):
                            re = T.simd.vld(e[i, group * 64])
                            acc[0] = T.simd.vadd(acc[0], re, mask)
                        row_sum = T.simd.vcadd(acc[0], mask)
                        T.simd.vsts(sm[i], row_sum, one, "ONEPT_B32")

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        rsum = T.simd.vld(sm[i], "BRC_B32")
                        for group in T.serial(C // 64):
                            re = T.simd.vld(e[i, group * 64])
                            ry = T.simd.vdiv(re, rsum, mask)
                            T.simd.vsts(y[i, group * 64], ry, mask)

                T.copy(mx, Max[begin : begin + rows])
                T.copy(shifted, Shift[begin : begin + rows, 0:C])
                T.copy(e, Exp[begin : begin + rows, 0:C])
                T.copy(sm, Sum[begin : begin + rows])
                T.copy(y, Y[begin : begin + rows, 0:C])

    return tilelang.compile(main, out_idx=[1, 2, 3, 4, 5])
