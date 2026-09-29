"""C=128 row Softmax: retain all intermediate values in one SIMD VF."""

import tilelang
import tilelang.language as T


def softmax_fused(M=9216, rows=8, blocks=36, stages=2):
    if min(M, rows, blocks, stages) <= 0 or rows % 8 or M % (rows * blocks):
        raise ValueError("Require rows % 8 == 0 and M % (rows * blocks) == 0")
    if 2 * rows * 128 * 4 * stages > 248 * 1024:
        raise ValueError("Input/output buffers exceed the 248 KiB UB budget")

    @T.prim_func
    def main(
        A: T.Tensor((M, 128), "float32"),
        Y: T.Tensor((M, 128), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((rows, 128), "float32")
            y = T.alloc_shared((rows, 128), "float32")
            T.annotate_buffer_versions({a: stages, y: stages})

            for it in T.Pipelined(M // (rows * blocks), num_stages=stages):
                begin = bx * (M // blocks) + it * rows
                T.copy(A[begin : begin + rows, :], a)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in T.serial(rows):
                        x0 = T.simd.vld(a[i, 0])
                        x1 = T.simd.vld(a[i, 64])
                        # 1. Row maximum; broadcast lane 0 to every lane.
                        mx = T.simd.vcmax(T.simd.vmax(x0, x1, mask), mask)
                        mx_all = T.simd.vdupv(mx, mask)
                        # 2. Subtract maximum. 3. Exponentiate.
                        e0 = T.simd.vexp(T.simd.vsub(x0, mx_all, mask), mask)
                        e1 = T.simd.vexp(T.simd.vsub(x1, mx_all, mask), mask)
                        # 4. Row sum; broadcast it within the registers.
                        sm = T.simd.vcadd(T.simd.vadd(e0, e1, mask), mask)
                        sm_all = T.simd.vdupv(sm, mask)
                        # 5. Normalize; only the final result is stored in UB.
                        T.simd.vsts(y[i, 0], T.simd.vdiv(e0, sm_all, mask), mask)
                        T.simd.vsts(y[i, 64], T.simd.vdiv(e1, sm_all, mask), mask)

                T.copy(y, Y[begin : begin + rows, :])

    return tilelang.compile(main, out_idx=-1)
