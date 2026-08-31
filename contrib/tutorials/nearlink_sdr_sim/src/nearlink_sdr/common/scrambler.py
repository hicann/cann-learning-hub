"""信道比特加扰 -- TXS-10002-2025 标准 6.10.4

使用 7bit 左移 Galois LFSR, 生成多项式 x^7 + x^4 + 1。
对编码后的比特序列进行异或加扰, 用于数据白化。
"""

from __future__ import annotations

__all__ = [
    "broadcast_seed",
    "data_link_seed",
    "descramble",
    "scramble",
    "scramble_sequence",
]


import numpy as np
from numpy.typing import NDArray

# Galois LFSR 反馈掩码: x^4 + x^0 → 位 4 和位 0
_GALOIS_MASK = 0x11


def _lfsr_step(state: int) -> tuple[int, int]:
    """Galois LFSR 单步: 输出 1bit, 返回 (output_bit, new_state)。

    左移 Galois LFSR, 多项式 x^7 + x^4 + 1。
    输出位: 寄存器 6 (MSB)。
    反馈: 输出位异或到寄存器位 0 和位 4。
    """
    output = (state >> 6) & 1
    state = ((state << 1) & 0x7F) ^ (output * _GALOIS_MASK)
    return output, state


_PERIOD = 127  # 7-bit 原始多项式周期

# 预计算所有 128 个种子的完整 127 步输出序列
_SEED_CACHE: dict[int, NDArray[np.uint8]] = {}


def _precompute_period(seed: int) -> NDArray[np.uint8]:
    """计算种子对应的完整 LFSR 周期输出。"""
    if seed in _SEED_CACHE:
        return _SEED_CACHE[seed]
    period = _PERIOD if seed != 0 else 1
    out = np.empty(period, dtype=np.uint8)
    state = seed
    for i in range(period):
        out[i] = (state >> 6) & 1
        output = out[i]
        state = ((state << 1) & 0x7F) ^ (output * _GALOIS_MASK)
    _SEED_CACHE[seed] = out
    return out


def scramble_sequence(length: int, seed: int) -> NDArray[np.uint8]:
    """生成指定长度的加扰序列。

    :param length: 需要的加扰比特数。
    :type length: int
    :param seed: 7bit 初始种子 (0 ~ 127)。
    :type seed: int
    :returns: 加扰序列, 元素为 0 或 1。
    :rtype: NDArray[np.uint8]
    """
    if not 0 <= seed <= 127:
        raise ValueError(f"seed 必须在 0..127 范围内, 收到 {seed}")
    if length == 0:
        return np.empty(0, dtype=np.uint8)
    period_seq = _precompute_period(seed)
    period = len(period_seq)
    if length <= period:
        return period_seq[:length].copy()
    # 重复拼接
    repeats = length // period + 1
    return np.tile(period_seq, repeats)[:length].copy()


def scramble(bits: NDArray[np.uint8], seed: int) -> NDArray[np.uint8]:
    """对比特序列进行加扰 (XOR)。

    :param bits: 待加扰比特, 元素为 0 或 1。
    :type bits: NDArray[np.uint8]
    :param seed: 7bit 初始种子。
    :type seed: int
    :returns: 加扰后的比特序列。
    :rtype: NDArray[np.uint8]
    """
    seq = scramble_sequence(len(bits), seed)
    return (bits ^ seq).astype(np.uint8)


def descramble(bits: NDArray[np.uint8], seed: int) -> NDArray[np.uint8]:
    """解扰。加扰和解扰操作完全相同 (自逆性)。"""
    return scramble(bits, seed)


def broadcast_seed(physical_channel: int) -> int:
    """广播帧的加扰种子: 物理信道号。

    :param physical_channel: 物理信道号 (0..397)。取低 7 位。
    :type physical_channel: int
    :returns: 7bit 种子。
    :rtype: int
    """
    return physical_channel & 0x7F


def data_link_seed(slot_number: int) -> int:
    """数据链路的加扰种子。

    将事件起始时刻调度时隙序号的低 6 位设为寄存器 0..5,
    寄存器 6 设为 1。

    :param slot_number: 调度时隙序号。
    :type slot_number: int
    :returns: 7bit 种子。
    :rtype: int
    """
    return (1 << 6) | (slot_number & 0x3F)
