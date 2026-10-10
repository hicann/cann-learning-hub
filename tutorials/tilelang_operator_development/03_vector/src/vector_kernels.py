"""Chapter 3: finite, contiguous fp32; no tail tiles.
Derived from Ascend examples; reference implementations are in src/original.
"""

import tilelang
import tilelang.ascend.language as T


def check_config(N, tile, blocks, versions=1, buffers=3):
    if min(N, tile, blocks, versions) <= 0 or tile % 64 or N % (tile * blocks):
        raise ValueError("Require positive sizes, tile % 64 == 0 and N % (tile*blocks) == 0")
    if buffers * tile * 4 * versions > 248 * 1024:
        raise ValueError("Local workspaces exceed this backend's 248 KiB SimdVF budget")


def add_serial(N, tile=1024, blocks=1):
    check_config(N, tile, blocks, 1, 3)

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        B: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")

            for it in T.serial(N // (blocks * tile)):
                begin = (it * blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for group in T.serial(tile // 64):
                        ra = T.simd.vld(a[group * 64])
                        rb = T.simd.vld(b[group * 64])
                        rc = T.simd.vadd(ra, rb, mask)
                        T.simd.vsts(c[group * 64], rc, mask)

                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, out_idx=-1)


def add_pipeline(N, tile=1024, blocks=1, stages=2):
    check_config(N, tile, blocks, stages, 3)

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        B: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: stages, b: stages, c: stages})

            for it in T.Pipelined(N // (blocks * tile), num_stages=stages):
                begin = (it * blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for group in T.serial(tile // 64):
                        ra = T.simd.vld(a[group * 64])
                        rb = T.simd.vld(b[group * 64])
                        rc = T.simd.vadd(ra, rb, mask)
                        T.simd.vsts(c[group * 64], rc, mask)

                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, out_idx=-1)


def unary_serial(N, tile=1024, blocks=1, op='relu'):
    check_config(N, tile, blocks, 1, 2)
    if op not in ("relu", "exp"):
        raise ValueError("op must be relu or exp")

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")

            for it in T.serial(N // (blocks * tile)):
                begin = bx * (N // blocks) + it * tile
                T.copy(A[begin : begin + tile], a)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    zero = T.simd.vdup(0.0, "float32", mask)
                    for group in T.serial(tile // 64):
                        ra = T.simd.vld(a[group * 64])
                        if op == "relu":
                            rc = T.simd.vmax(ra, zero, mask)
                        else:
                            rc = T.simd.vexp(ra, mask)
                        T.simd.vsts(c[group * 64], rc, mask)

                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, out_idx=-1)


def unary_pipeline(N, tile=1024, blocks=1, op='relu', stages=2):
    check_config(N, tile, blocks, stages, 2)
    if op not in ("relu", "exp"):
        raise ValueError("op must be relu or exp")

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: stages, c: stages})

            for it in T.Pipelined(N // (blocks * tile), num_stages=stages):
                begin = bx * (N // blocks) + it * tile
                T.copy(A[begin : begin + tile], a)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    zero = T.simd.vdup(0.0, "float32", mask)
                    for group in T.serial(tile // 64):
                        ra = T.simd.vld(a[group * 64])
                        if op == "relu":
                            rc = T.simd.vmax(ra, zero, mask)
                        else:
                            rc = T.simd.vexp(ra, mask)
                        T.simd.vsts(c[group * 64], rc, mask)

                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, out_idx=-1)


def add_simt(N=64):
    check_config(N, N, 1)

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        B: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(1) as bx:
            a = T.alloc_shared((N,), 'float32')
            b = T.alloc_shared((N,), 'float32')
            c = T.alloc_shared((N,), 'float32')
            T.copy(A, a)
            T.copy(B, b)

            with T.SimtVF(threads=128):
                for i in T.Parallel(N):
                    c[i] = a[i] + b[i]
            T.copy(c, C)

    return tilelang.compile(main, out_idx=-1)


def binary_instructions(op='add', N=64):
    check_config(N, N, 1)
    if op not in ('add', 'mul'):
        raise ValueError('Unsupported instruction exercise')

    @T.prim_func
    def main(
        A: T.Tensor((N,), 'float32'),
        B: T.Tensor((N,), 'float32'),
        C: T.Tensor((N,), 'float32'),
    ):
        with T.Kernel(1) as bx:
            a = T.alloc_shared((N,), 'float32')
            b = T.alloc_shared((N,), 'float32')
            c = T.alloc_shared((N,), 'float32')
            T.copy(A, a)
            T.copy(B, b)

            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in T.serial(N // 64):
                    ra = T.simd.vld(a[i * 64])
                    rb = T.simd.vld(b[i * 64])
                    if op == 'add':
                        rc = T.simd.vadd(ra, rb, mask)
                    else:
                        rc = T.simd.vmul(ra, rb, mask)
                    T.simd.vsts(c[i * 64], rc, mask)

            T.copy(c, C)

    return tilelang.compile(main, out_idx=-1)
