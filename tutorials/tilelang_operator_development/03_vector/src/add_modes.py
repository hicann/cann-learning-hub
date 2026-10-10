"""Same Add workload, tiling and interface for SimdVF/SimtVF comparison."""

import tilelang
import tilelang.ascend.language as T


def check_add_config(N, tile, blocks, mode):
    if mode not in ("simd", "simt"):
        raise ValueError("mode must be simd or simt")
    if min(N, tile, blocks) <= 0 or tile % 64 or N % (tile * blocks):
        raise ValueError("Require tile % 64 == 0 and N % (tile * blocks) == 0")
    if mode == "simd" and 3 * tile * 4 > 248 * 1024:
        raise ValueError("SimdVF three fp32 buffers exceed the 248 KiB UB limit")


def add_simd(N=64, tile=64, blocks=1):
    check_add_config(N, tile, blocks, "simd")

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")

            for it in T.serial(N // (blocks * tile)):
                begin = bx * (N // blocks) + it * tile
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


def add_simt(N=64, tile=64, blocks=1):
    check_add_config(N, tile, blocks, "simt")

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            begin = bx * (N // blocks)

            with T.SimtVF(threads=2048):
                for i in T.Parallel(N // blocks):
                    C[begin + i] = A[begin + i] + B[begin + i]

    return tilelang.compile(main, out_idx=-1)
