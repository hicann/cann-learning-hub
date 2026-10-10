"""Stable float32 row softmax on AscendC using SimdVF.

Each logical task processes one complete input row.
"""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch


def _validate_shape(rows: int, cols: int) -> None:
    if type(rows) is not int or not 4 <= rows <= 256 or rows % 4:
        raise ValueError('rows must be an integer in [4, 256] and a multiple of 4')
    if type(cols) is not int or not 128 <= cols <= 2048 or cols % 64:
        raise ValueError('cols must be an integer in [128, 2048] and a multiple of 64')


@lru_cache(maxsize=128)
def build(rows: int, cols: int):
    """Compile a callable accepting validated contiguous float32 NPU X and Y."""
    _validate_shape(rows, cols)

    @T.prim_func
    def row_softmax_kernel(X: T.Tensor((rows, cols), 'float32'), Y: T.Tensor((rows, cols), 'float32')):
        with T.Kernel(rows) as bx:
            x = T.alloc_shared((1, cols), 'float32')
            e = T.alloc_shared((1, cols), 'float32')
            y = T.alloc_shared((1, cols), 'float32')
            m = T.alloc_shared((1,), 'float32')
            s = T.alloc_shared((1,), 'float32')
            T.copy(X[bx : bx + 1, :], x)
            with T.SimdVF():
                T.reduce_max(x, m, dim=1, clear=True)
            with T.SimdVF():
                for i, j in T.Parallel(1, cols):
                    e[i, j] = T.exp(x[i, j] - m[i])
            with T.SimdVF():
                T.reduce_sum(e, s, dim=1, clear=True)
            with T.SimdVF():
                for i, j in T.Parallel(1, cols):
                    y[i, j] = e[i, j] / s[i]
            T.copy(y, Y[bx : bx + 1, :])

    return tilelang.compile(row_softmax_kernel, target='ascend', execution_backend='tvm_ffi')


def run(x: torch.Tensor) -> torch.Tensor:
    """Return a new contiguous softmax tensor without modifying x.

    x must be a contiguous strided float32 NPU matrix with 4 <= M <= 256,
    M % 4 == 0, 128 <= N <= 2048, N % 64 == 0, and finite values in [-80, 80].
    Raises TypeError for an invalid dtype and ValueError for other domain errors.
    This interface provides forward computation only.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError('x must be a tensor')
    if x.dtype != torch.float32:
        raise TypeError('x dtype must be float32')
    if x.ndim != 2:
        raise ValueError('x rank must be 2')
    _validate_shape(*x.shape)
    if x.layout != torch.strided:
        raise ValueError('x must have strided layout')
    if not x.is_contiguous():
        raise ValueError('x must be contiguous')
    if x.device.type != 'npu':
        raise ValueError('x device must be Ascend NPU')
    if not torch.isfinite(x).all().item():
        raise ValueError('x values must be finite')
    if (x.abs() > 80).any().item():
        raise ValueError('x values must have absolute value <= 80')
    with torch.npu.device(x.device):
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        build(*x.shape)(x, y)
    return y
