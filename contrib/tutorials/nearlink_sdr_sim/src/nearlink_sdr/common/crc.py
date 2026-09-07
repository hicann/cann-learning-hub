
__all__ = [
    "CRC12_POLY",
    "CRC24A_POLY",
    "CRC24B_POLY",
    "CRC32_POLY",
    "crc_attach",
    "crc_calculate",
    "crc_check",
]


import numpy as np

try:
    from nearlink_sdr_accel import rust_crc_calculate as _rust_crc
    _HAS_RUST_CRC = True
except ImportError:
    _HAS_RUST_CRC = False

# TXS-10002-2025 6.10.1 循环冗余校验
# 标准定义的CRC生成多项式（用比特位表示，高位在前）

# CRC12: D^12 + D^11 + D^3 + D^2 + D + 1
CRC12_POLY = 0x80F

# CRC24A: D^24 + D^10 + D^9 + D^6 + D^4 + D^3 + D + 1
CRC24A_POLY = 0x00065B

# CRC24B: D^24 + D^23 + D^21 + D^20 + D^17 + D^15 + D^13 + D^12 + D^8 + D^4 + D^2 + D + 1
CRC24B_POLY = 0xB2B117

# CRC32: D^32 + D^26 + D^23 + D^22 + D^16 + D^12 + D^11 + D^10 + D^8 + D^7 + D^5 + D^4 + D^2 + D + 1
CRC32_POLY = 0x04C11DB7


def crc_calculate(data_bits: np.ndarray, poly: int, crc_len: int,
                  seed: int = 0) -> np.ndarray:
    """按照 TXS-10002-2025 6.10.1 计算循环冗余校验。

    :param data_bits: 输入信息比特序列, shape (A,), 值为 0/1
    :param poly: CRC生成多项式（不含最高位 D^L）
    :param crc_len: 校验比特长度 L
    :param seed: CRC生成种子

    :returns: 校验比特序列, shape (L,), 值为 0/1
    """
    if _HAS_RUST_CRC:
        return np.asarray(_rust_crc(np.asarray(data_bits, dtype=np.int64),
                                    poly, crc_len, seed))

    # 初始化移位寄存器
    reg = seed & ((1 << crc_len) - 1)

    for bit in data_bits:
        # 取最高位
        msb = (reg >> (crc_len - 1)) & 1
        feedback = int(bit) ^ msb
        reg = (reg << 1) & ((1 << crc_len) - 1)
        if feedback:
            reg ^= poly

    # 将寄存器内容转成比特数组（高位在前）
    parity = np.zeros(crc_len, dtype=int)
    for i in range(crc_len):
        parity[i] = (reg >> (crc_len - 1 - i)) & 1
    return parity


def crc_attach(data_bits: np.ndarray, poly: int, crc_len: int,
               seed: int = 0, mask: np.ndarray | None = None) -> np.ndarray:
    """对输入信息比特序列附加CRC校验比特。

    :returns: b_0, b_1, ..., b_{B-1}, 其中 B = A + L
    """
    parity = crc_calculate(data_bits, poly, crc_len, seed)
    if mask is not None:
        parity = (parity + mask[:crc_len]) % 2
    return np.concatenate([data_bits, parity])


def crc_check(received_bits: np.ndarray, poly: int, crc_len: int,
              seed: int = 0, mask: np.ndarray | None = None) -> bool:
    """对接收比特序列进行CRC校验。

    :returns: True 表示校验通过（无错误），False 表示校验失败
    """
    data_bits = received_bits[:-crc_len]
    rx_parity = received_bits[-crc_len:].copy()
    if mask is not None:
        rx_parity = (rx_parity + mask[:crc_len]) % 2
    calc_parity = crc_calculate(data_bits, poly, crc_len, seed)
    return np.array_equal(rx_parity, calc_parity)
