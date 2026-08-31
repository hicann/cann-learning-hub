
__all__ = [
    "GFSK_BW_CONFIG",
    "GFSKDemodulator",
    "GFSKModulator",
]


import numpy as np

# TXS-10002-2025 6.2.1.1 GFSK调制
# BT = 0.5, 调制系数 h 在 0.45~0.55 之间

# 标准带宽与符号间隔对应关系
GFSK_BW_CONFIG = {
    0.1:  {"symbol_period_us": 10,  "min_freq_dev_khz": 23.125},
    0.125: {"symbol_period_us": 8,  "min_freq_dev_khz": 46.25},
    0.25: {"symbol_period_us": 4,   "min_freq_dev_khz": 92.5},
    0.5:  {"symbol_period_us": 2,   "min_freq_dev_khz": 185},
    1.0:  {"symbol_period_us": 1,   "min_freq_dev_khz": 185},
    2.0:  {"symbol_period_us": 0.5, "min_freq_dev_khz": 370},
    4.0:  {"symbol_period_us": 0.25, "min_freq_dev_khz": 740},
}


def _gaussian_filter(bt: float, span: int, sps: int) -> np.ndarray:
    """生成高斯滤波器的脉冲响应。"""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = np.sqrt(2 * np.pi) / alpha * np.exp(-2 * (np.pi * t / alpha) ** 2)
    h = h / np.sum(h)
    return h


class GFSKModulator:
    """TXS-10002-2025 6.2.1.1 GFSK调制器。

    比特1表示正频偏，比特0表示负频偏。
    """

    def __init__(self, sps: int = 8, mod_index: float = 0.5,
                 bt: float = 0.5, gauss_span: int = 3):
        if not 0.45 <= mod_index <= 0.55:
            raise ValueError(
                f"mod_index={mod_index} 超出标准范围 [0.45, 0.55] (§6.2.1.1)"
            )
        self.sps = sps
        self.mod_index = mod_index
        self.bt = bt
        self.gauss_span = gauss_span
        self._gauss_filter = _gaussian_filter(bt, gauss_span, sps)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """GFSK调制。

        :param bits: 输入比特序列, shape (N,), 值为 0/1

        :returns: 复基带IQ信号, shape (N*sps,)
        """
        # NRZ映射: 0 -> -1, 1 -> +1
        nrz = 2.0 * bits.astype(float) - 1.0

        # 上采样：使用矩形脉冲（repeat），而非冲激
        upsampled = np.repeat(nrz, self.sps)

        # 高斯滤波平滑频率轨迹
        filtered = np.convolve(upsampled, self._gauss_filter, mode='same')

        # 频率积分得到瞬时相位
        # 每个符号的总相位变化应为 h*pi
        freq_deviation = self.mod_index * np.pi / self.sps
        phase = np.cumsum(filtered) * freq_deviation

        # 生成复基带信号
        signal = np.exp(1j * phase)
        return signal


class GFSKDemodulator:
    """GFSK非相干解调器（基于频率鉴别器）。"""

    def __init__(self, sps: int = 8):
        self.sps = sps

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """GFSK解调。

        :param signal: 复基带IQ信号

        :returns: 解调后的比特序列, 值为 0/1
        """
        # 频率鉴别：取相邻采样点的相位差
        phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))

        # 按符号累积频率 (包含末尾不足一个完整符号的部分)
        n_symbols = -(-len(phase_diff) // self.sps)  # ceiling division
        bits = np.zeros(n_symbols, dtype=int)
        for i in range(n_symbols):
            start = i * self.sps
            end = min(start + self.sps, len(phase_diff))
            segment = phase_diff[start:end]
            bits[i] = 1 if np.sum(segment) > 0 else 0

        return bits
