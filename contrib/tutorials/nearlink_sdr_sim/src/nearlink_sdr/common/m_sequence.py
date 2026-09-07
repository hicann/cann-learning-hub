
__all__ = [
    "M31_CONFIGS",
    "M63_CONFIGS",
    "generate_m_sequence",
    "m31_sequence",
    "m63_sequence",
]


import numpy as np

try:
    from nearlink_sdr_accel import rust_generate_m_sequence as _rust_m_seq
    _HAS_RUST_MSEQ = True
except ImportError:
    _HAS_RUST_MSEQ = False

# TXS-10002-2025 6.10.2 m序列
# 反馈系数与移位寄存器初始值定义

# 31长m序列（5阶LFSR）
M31_CONFIGS = {
    0: {"taps": 0b100101, "init": 0b00001},
    1: {"taps": 0b101001, "init": 0b00001},
    2: {"taps": 0b110111, "init": 0b00001},
    3: {"taps": 0b111011, "init": 0b00001},
    4: {"taps": 0b111101, "init": 0b00001},
    5: {"taps": 0b101111, "init": 0b00001},
}

# 63长m序列（6阶LFSR）
M63_CONFIGS = {
    0: {"taps": 0b1000011, "init": 0b000001},
    1: {"taps": 0b1100001, "init": 0b000001},
    2: {"taps": 0b1100111, "init": 0b000001},
    3: {"taps": 0b1110011, "init": 0b000001},
    4: {"taps": 0b1101101, "init": 0b000001},
    5: {"taps": 0b1011011, "init": 0b000001},
}


def generate_m_sequence(order: int, taps: int, init_val: int,
                        length: int | None = None) -> np.ndarray:
    """生成线性反馈移位寄存器产生的m序列。

    :param order: LFSR阶数（移位寄存器位数）
    :param taps: 反馈系数（二进制表示，包含最高位）
    :param init_val: 移位寄存器初始值
    :param length: 输出序列长度，默认为 2^order - 1（完整m序列周期）

    :returns: m序列比特数组, shape (length,), 值为 0/1
    """
    if length is None:
        length = (1 << order) - 1

    if _HAS_RUST_MSEQ:
        return np.asarray(_rust_m_seq(order, taps, init_val, length))

    reg = init_val
    mask = (1 << order) - 1
    seq = np.zeros(length, dtype=int)

    for i in range(length):
        # 输出最低位
        seq[i] = reg & 1

        # 计算反馈：将reg与taps去掉最高位后做AND，然后求各位异或
        feedback_bits = reg & (taps & mask)
        feedback = 0
        for _ in range(order):
            feedback ^= feedback_bits & 1
            feedback_bits >>= 1

        # 移位
        reg = ((reg >> 1) | (feedback << (order - 1))) & mask

    return seq


def m31_sequence(index: int, length: int = 31) -> np.ndarray:
    """生成标准中定义的31长m序列。

    :param index: 序列编号 0~5
    :param length: 输出长度，默认31
    """
    cfg = M31_CONFIGS[index]
    return generate_m_sequence(5, cfg["taps"], cfg["init"], length)


def m63_sequence(index: int, length: int = 63) -> np.ndarray:
    """生成标准中定义的63长m序列。

    :param index: 序列编号 0~5
    :param length: 输出长度，默认63
    """
    cfg = M63_CONFIGS[index]
    return generate_m_sequence(6, cfg["taps"], cfg["init"], length)
