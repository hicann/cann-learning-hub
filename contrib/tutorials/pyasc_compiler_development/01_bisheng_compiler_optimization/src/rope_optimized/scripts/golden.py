# ============================================================================
# Golden 计算（双通路共用）- RoPE Optimized
# ============================================================================
#
# 实现 HuggingFace 风格的 RoPE：
#   output = x * cos + rotate_half(x) * sin
#
# 其中 rotate_half(x) 将 x 沿最后一维分成两半 [x1, x2]，返回 [-x2, x1]
#
# 本文件被 gen_data.py 和 test_torch.py 共同引用。

import numpy as np
import torch


def rotate_half(x):
    """将 x 沿最后一维分成两半，返回 [-x2, x1]

    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    return cat([-x2, x1], dim=-1)
    """
    if isinstance(x, np.ndarray):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return np.concatenate([-x2, x1], axis=-1)
    else:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)


def compute_golden(x, cos, sin):
    """计算 RoPE 的参考输出。

    Args:
        x: [B, S, H, D] numpy array 或 torch.Tensor
        cos: [S, D] numpy array 或 torch.Tensor
        sin: [S, D] numpy array 或 torch.Tensor

    Returns:
        [B, S, H, D] 参考输出
    """
    if isinstance(x, np.ndarray):
        # 广播 cos/sin 从 [S, D] 到 [B, S, H, D]
        cos_expanded = cos[np.newaxis, :, np.newaxis, :]  # [1, S, 1, D]
        sin_expanded = sin[np.newaxis, :, np.newaxis, :]  # [1, S, 1, D]
    else:
        cos_expanded = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
        sin_expanded = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]

    output = x * cos_expanded + rotate_half(x) * sin_expanded
    return output
