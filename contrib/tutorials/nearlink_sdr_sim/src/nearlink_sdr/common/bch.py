
__all__ = [
    "BCH_31_26_GEN",
    "BCH_63_24_GEN",
    "bch_31_26_encode",
    "bch_63_24_encode",
]


import numpy as np

# TXS-10002-2025 6.2.3.1 / 6.2.3.2 BCH编码器


def _gf2_poly_mod(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """GF(2)多项式模运算。

    多项式用比特数组表示，索引i对应d^i的系数。
    """
    r = dividend.copy()
    deg = len(divisor) - 1
    for i in range(len(r) - 1, deg - 1, -1):
        if r[i]:
            r[i - deg:i + 1] ^= divisor
    return r[:deg]


# BCH(31,26) 生成多项式: G(d) = 0045 (八进制) = x^5 + x^2 + 1
BCH_31_26_GEN = np.array([1, 0, 1, 0, 0, 1], dtype=int)

# BCH(63,24) 生成多项式: 通过GF(2^6)最小多项式乘积计算得出
# 本原多项式 x^6+x+1, 连续根 α^1..α^14, 生成多项式阶数39
BCH_63_24_GEN = np.array([
    1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
], dtype=int)


def bch_31_26_encode(info_26: np.ndarray) -> np.ndarray:
    """BCH(31,26)系统编码。

    按标准6.2.3.1:
        C(d) = d^5 * Ã(d) mod G(d)
        S(d) = Ã(d) + d^26 * C(d)

    :param info_26: 26比特信息序列（Ã(d)的系数，索引0=d^0）

    :returns: 31比特码字
    """
    g = BCH_31_26_GEN  # 阶数5
    n_parity = 5

    # d^5 * Ã(d)
    shifted = np.zeros(26 + n_parity, dtype=int)
    shifted[n_parity:] = info_26

    parity = _gf2_poly_mod(shifted, g)

    codeword = np.zeros(31, dtype=int)
    codeword[:26] = info_26
    codeword[26:31] = parity
    return codeword


def bch_63_24_encode(info_24: np.ndarray) -> np.ndarray:
    """BCH(63,24)系统编码。

    按标准6.2.3.2:
        C(d) = d^39 * A(d) mod G(d)
        S(d) = A(d) + d^24 * C(d)

    :param info_24: 24比特信息序列（A(d)的系数，索引0=d^0）

    :returns: 63比特码字
    """
    g = BCH_63_24_GEN  # 阶数39
    n_parity = 39

    # d^39 * A(d)
    shifted = np.zeros(24 + n_parity, dtype=int)
    shifted[n_parity:] = info_24

    parity = _gf2_poly_mod(shifted, g)

    codeword = np.zeros(63, dtype=int)
    codeword[:24] = info_24
    codeword[24:63] = parity
    return codeword
