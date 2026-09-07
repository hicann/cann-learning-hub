"""伪随机比特序列发生器 -- TXS-10002-2025 标准 8.3.5 / 14.1

PRBS11: x^11 + x^2 + 1, 周期 2047, 用于测试有用信号数据。
PRBS17: x^17 + x^3 + 1, 周期 131071, 用于测试干扰信号数据。
"""

from __future__ import annotations

__all__ = [
    "prbs11",
    "prbs17",
]


import numpy as np
from numpy.typing import NDArray


def _lfsr_generate(state: list[int], tap_a: int, tap_b: int,
                   length: int) -> NDArray[np.uint8]:
    """通用 LFSR 序列生成。

    结构: 左移 LFSR, 输出端为最高位 stage, 反馈: stage[tap_a] XOR stage[tap_b] → stage[0]。

    :param state: 初始状态, state[0] 为输入端, state[-1] 为输出端。
    :type state: list[int]
    :param tap_a: 反馈抓头位置 (stage 编号)。
    :type tap_a: int
    :param tap_b: 反馈抓头位置 (stage 编号)。
    :type tap_b: int
    :param length: 生成序列长度。
    :type length: int
    :returns: 输出比特序列。
    :rtype: NDArray[np.uint8]
    """
    n = len(state)
    out = np.empty(length, dtype=np.uint8)
    s = list(state)
    for i in range(length):
        out[i] = s[n - 1]
        feedback = s[tap_a] ^ s[tap_b]
        for j in range(n - 1, 0, -1):
            s[j] = s[j - 1]
        s[0] = feedback
    return out


def prbs11(length: int, seed: int = 0x7FF) -> NDArray[np.uint8]:
    """PRBS11 序列生成。

    多项式 x^11 + x^2 + 1。输出端 stage 10, 反馈 stage[10] XOR stage[1] → stage[0]。

    :param length: 输出比特数。
    :type length: int
    :param seed: 11 位初始种子 (默认全 1)。
    :type seed: int
    :returns: PRBS11 序列。
    :rtype: NDArray[np.uint8]
    """
    state = [(seed >> i) & 1 for i in range(11)]
    return _lfsr_generate(state, 10, 1, length)


def prbs17(length: int, seed: int = 0x1FFFF) -> NDArray[np.uint8]:
    """PRBS17 序列生成。

    多项式 x^17 + x^3 + 1。输出端 stage 16, 反馈 stage[16] XOR stage[2] → stage[0]。

    :param length: 输出比特数。
    :type length: int
    :param seed: 17 位初始种子 (默认全 1)。
    :type seed: int
    :returns: PRBS17 序列。
    :rtype: NDArray[np.uint8]
    """
    state = [(seed >> i) & 1 for i in range(17)]
    return _lfsr_generate(state, 16, 2, length)
