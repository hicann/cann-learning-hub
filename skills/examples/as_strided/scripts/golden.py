"""Golden implementation for as_strided operator."""
import numpy as np


def as_strided_golden(input_x, size, stride, storage_offset):
    """
    CPU golden: output[i_0,...,i_{n-1}] = input[storage_offset + sum(i_j * stride_j)]
    
    Args:
        input_x: numpy array, input tensor
        size: list/tuple of ints, output shape
        stride: list/tuple of ints, stride for each dimension
        storage_offset: int, starting offset in the input
    
    Returns:
        numpy array with shape=size, dtype same as input_x
    """
    output = np.zeros(size, dtype=input_x.dtype)
    input_flat = input_x.flatten()
    for idx in np.ndindex(*size):
        src_idx = storage_offset + sum(i * s for i, s in zip(idx, stride))
        # Clamp for safety (matches kernel defensive clamp)
        src_idx = max(0, min(src_idx, len(input_flat) - 1))
        output[idx] = input_flat[src_idx]
    return output
