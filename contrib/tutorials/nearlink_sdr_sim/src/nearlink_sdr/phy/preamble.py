
__all__ = [
    "BW_SYMBOL_RATE",
    "PREAMBLE_CONFIG",
    "generate_preamble",
    "gfsk_preamble_bits",
    "psk_preamble_phases",
    "psk_preamble_symbols",
]


import numpy as np

# TXS-10002-2025 6.2.2 前导信号

# 各带宽对应的符号速率 (MHz)
BW_SYMBOL_RATE = {
    0.1: 0.1,
    0.125: 0.125,
    0.25: 0.25,
    0.5: 0.5,
    1.0: 1.0,
    2.0: 2.0,
    4.0: 4.0,
}


def gfsk_preamble_bits(symbol_rate_mhz: float = 1.0,
                       duration_us: float = 10.0) -> np.ndarray:
    """生成GFSK前导码比特序列。

    帧类型1使用 [0,1] 交替序列，第一个码元为0。

    :param symbol_rate_mhz: 符号速率 (MHz)
    :param duration_us: 前导码持续时间 (μs)

    :returns: 前导码比特序列, 值为 0/1
    """
    n_symbols = int(symbol_rate_mhz * duration_us)
    bits = np.zeros(n_symbols, dtype=int)
    bits[1::2] = 1  # [0,1,0,1,...]
    return bits


def psk_preamble_phases(symbol_rate_mhz: float = 1.0,
                        duration_us: float = 10.0) -> np.ndarray:
    """生成PSK前导码相位序列。

    帧类型2/3/4使用 [π/4, 0] 交替相位。

    :param symbol_rate_mhz: 符号速率 (MHz)
    :param duration_us: 前导码持续时间 (μs)，帧类型2=10，帧类型3=12，帧类型4=16

    :returns: 前导码相位序列 (弧度)
    """
    n_symbols = int(symbol_rate_mhz * duration_us)
    phases = np.zeros(n_symbols, dtype=float)
    phases[0::2] = np.pi / 4  # [π/4, 0, π/4, 0, ...]
    return phases


def psk_preamble_symbols(symbol_rate_mhz: float = 1.0,
                         duration_us: float = 10.0) -> np.ndarray:
    """生成PSK前导码复数符号序列。

    :param symbol_rate_mhz: 符号速率 (MHz)
    :param duration_us: 前导码持续时间 (μs)

    :returns: 前导码复数符号, shape (N,)
    """
    phases = psk_preamble_phases(symbol_rate_mhz, duration_us)
    return np.exp(1j * phases)


# 前导码配置: {帧类型: (调制方式, 持续时间μs)}
PREAMBLE_CONFIG = {
    1: ("GFSK", 10.0),
    2: ("PSK", 10.0),
    3: ("PSK", 12.0),
    4: ("PSK", 16.0),
}


def generate_preamble(frame_type: int,
                      symbol_rate_mhz: float = 1.0):
    """生成指定帧类型的前导码。

    :param frame_type: 帧类型 1~4
    :param symbol_rate_mhz: 符号速率 (MHz)

    :returns: 帧类型1: 比特数组 (int ndarray)
        帧类型2/3/4: 复数符号数组 (complex ndarray)
    """
    mod_type, duration = PREAMBLE_CONFIG[frame_type]
    if mod_type == "GFSK":
        return gfsk_preamble_bits(symbol_rate_mhz, duration)
    else:
        return psk_preamble_symbols(symbol_rate_mhz, duration)
