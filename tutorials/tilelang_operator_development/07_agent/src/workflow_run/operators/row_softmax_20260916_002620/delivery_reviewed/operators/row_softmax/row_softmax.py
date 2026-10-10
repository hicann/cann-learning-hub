"""Stable float32 row softmax on AscendC using SimdVF.

Each logical task processes one complete input row.
"""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
import torch


def _validate_shape(rows: int, cols: int) -> None:
    if type(rows) is not int or not 4 <= rows <= 256 or rows % 4:
        raise ValueError('rows must be an integer in [4, 256] and a multiple of 4')
    if type(cols) is not int or not 128 <= cols <= 2048 or cols % 64:
        raise ValueError('cols must be an integer in [128, 2048] and a multiple of 64')


def build(rows: int, cols: int):
    """Compile a callable accepting validated contiguous float32 NPU X and Y."""
    _validate_shape(rows, cols)
    return _compile(rows, cols)


@lru_cache(maxsize=128)
def _compile(rows: int, cols: int):

    @T.prim_func
    def row_softmax_kernel(X: T.Tensor((rows, cols), 'float32'), Y: T.Tensor((rows, cols), 'float32')):
        with T.Kernel(rows) as bx:
            x = T.alloc_shared((1, cols), 'float32')
            y = T.alloc_shared((1, cols), 'float32')
            T.copy(X[bx : bx + 1, :], x)
            with T.SimdVF():
                mask = S.pset(32, 'PAT_ALL')
                lane_max = S.alloc_var('float32')
                lane_sum = S.alloc_var('float32')
                lane_max = S.vld(x[0, 0])
                for k in T.serial(1, cols // 64):
                    lane_max = S.vmax(lane_max, S.vld(x[0, k * 64]), mask)
                row_max = S.vdupv(S.vcmax(lane_max, mask), mask)
                lane_sum = S.vdup(0.0, 'float32', mask)
                for k in T.serial(cols // 64):
                    value = S.vexp(S.vsub(S.vld(x[0, k * 64]), row_max, mask), mask)
                    lane_sum = S.vadd(lane_sum, value, mask)
                row_sum = S.vdupv(S.vcadd(lane_sum, mask), mask)
                reciprocal = S.vdiv(S.vdup(1.0, 'float32', mask), row_sum, mask)
                for k in T.serial(cols // 64):
                    value = S.vexp(S.vsub(S.vld(x[0, k * 64]), row_max, mask), mask)
                    S.vsts(y[0, k * 64], S.vmul(value, reciprocal, mask), mask)
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
