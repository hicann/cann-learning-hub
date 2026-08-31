
__all__ = [
    "BPSK_MAP",
    "BPSK_NOROT_MAP",
    "BPSK_ROTATION",
    "QPSK_MAP",
    "QPSK_ROTATION",
    "SLE_RRC_BETA",
    "PSKDemodulator",
    "PSKModulator",
    "rrc_filter",
]


from functools import lru_cache

import numpy as np

# TXS-10002-2025 6.2.1.2 PSK调制
# 滚降系数 beta = 0.4

SLE_RRC_BETA = 0.4


@lru_cache(maxsize=8)
def rrc_filter(beta: float, sps: int, span: int = 10) -> np.ndarray:
    """生成平方根升余弦脉冲成型滤波器。

    :param beta: 滚降系数
    :param sps: 每符号采样数
    :param span: 单侧符号跨度

    :returns: 滤波器系数, 归一化使得匹配滤波后采样点增益为1
    """
    N = 2 * span * sps + 1
    t = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1) / sps

    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    # 归一化
    h = h / np.sqrt(np.sum(h ** 2))
    return h


# 标准中的PSK星座映射表

# BPSK: 0->90°, 1->-90°, 偶数符号额外顺时针旋转90°
BPSK_MAP = {
    0: np.exp(1j * np.pi / 2),
    1: np.exp(-1j * np.pi / 2),
}
BPSK_ROTATION = np.exp(-1j * np.pi / 2)  # 偶数位顺时针90°

# QPSK: 00->45°, 01->135°, 11->-135°, 10->-45°
QPSK_MAP = {
    0b00: np.exp(1j * np.pi / 4),
    0b01: np.exp(1j * 3 * np.pi / 4),
    0b11: np.exp(-1j * 3 * np.pi / 4),
    0b10: np.exp(-1j * np.pi / 4),
}
QPSK_ROTATION = np.exp(-1j * np.pi / 4)  # 偶数位顺时针45°

# 8PSK: 000->22.5°, 001->67.5°, 011->112.5°, 010->157.5°,
#        110->-157.5°, 111->-112.5°, 101->-67.5°, 100->-22.5°
_8PSK_MAP = {
    0b000: np.exp(1j * np.deg2rad(22.5)),
    0b001: np.exp(1j * np.deg2rad(67.5)),
    0b011: np.exp(1j * np.deg2rad(112.5)),
    0b010: np.exp(1j * np.deg2rad(157.5)),
    0b110: np.exp(-1j * np.deg2rad(157.5)),
    0b111: np.exp(-1j * np.deg2rad(112.5)),
    0b101: np.exp(-1j * np.deg2rad(67.5)),
    0b100: np.exp(-1j * np.deg2rad(22.5)),
}
_8PSK_ROTATION = np.exp(-1j * np.deg2rad(22.5))  # 偶数位顺时针22.5°

# 无相位旋转BPSK: 0->0°, 1->180°
BPSK_NOROT_MAP = {
    0: np.exp(1j * 0),
    1: np.exp(1j * np.pi),
}


class PSKModulator:
    """TXS-10002-2025 6.2.1.2 PSK调制器。

    支持 BPSK, QPSK, 8PSK 及无相位旋转BPSK。
    """

    def __init__(self, mod_type: str = "QPSK", sps: int = 4,
                 beta: float = SLE_RRC_BETA, rrc_span: int = 10):
        self.mod_type = mod_type.upper()
        self.sps = sps
        self.beta = beta
        self.rrc_span = rrc_span
        self._rrc = rrc_filter(beta, sps, rrc_span)

        if self.mod_type == "BPSK":
            self._map = BPSK_MAP
            self._bits_per_symbol = 1
            self._rotation = BPSK_ROTATION
        elif self.mod_type == "QPSK":
            self._map = QPSK_MAP
            self._bits_per_symbol = 2
            self._rotation = QPSK_ROTATION
        elif self.mod_type == "8PSK":
            self._map = _8PSK_MAP
            self._bits_per_symbol = 3
            self._rotation = _8PSK_ROTATION
        elif self.mod_type == "BPSK_NOROT":
            self._map = BPSK_NOROT_MAP
            self._bits_per_symbol = 1
            self._rotation = None
        else:
            raise ValueError(f"不支持的调制方式: {mod_type}")

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    def map_symbols(self, bits: np.ndarray) -> np.ndarray:
        """比特到星座符号的映射（含相位旋转）。"""
        bps = self._bits_per_symbol
        n_pad = (-len(bits)) % bps
        if n_pad:
            bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])

        n_symbols = len(bits) // bps
        symbols = np.zeros(n_symbols, dtype=complex)

        for i in range(n_symbols):
            idx = 0
            for b in range(bps):
                idx = (idx << 1) | int(bits[i * bps + b])
            symbols[i] = self._map[idx]
            # 偶数符号位额外旋转（标准中第一个符号处于奇数位，索引0为奇数位）
            if self._rotation is not None and (i % 2 == 1):
                symbols[i] *= self._rotation

        return symbols

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """PSK调制并进行RRC脉冲成型。

        :returns: 复基带IQ信号
        """
        symbols = self.map_symbols(bits)

        # 上采样
        upsampled = np.zeros(len(symbols) * self.sps, dtype=complex)
        upsampled[::self.sps] = symbols

        # RRC脉冲成型
        signal = np.convolve(upsampled, self._rrc, mode='same')
        return signal


class PSKDemodulator:
    """PSK解调器（匹配滤波+最近邻判决）。"""

    def __init__(self, mod_type: str = "QPSK", sps: int = 4,
                 beta: float = SLE_RRC_BETA, rrc_span: int = 10):
        self.mod_type = mod_type.upper()
        self.sps = sps
        self._rrc = rrc_filter(beta, sps, rrc_span)

        if self.mod_type == "BPSK":
            self._map = BPSK_MAP
            self._bits_per_symbol = 1
            self._rotation = BPSK_ROTATION
        elif self.mod_type == "QPSK":
            self._map = QPSK_MAP
            self._bits_per_symbol = 2
            self._rotation = QPSK_ROTATION
        elif self.mod_type == "8PSK":
            self._map = _8PSK_MAP
            self._bits_per_symbol = 3
            self._rotation = _8PSK_ROTATION
        elif self.mod_type == "BPSK_NOROT":
            self._map = BPSK_NOROT_MAP
            self._bits_per_symbol = 1
            self._rotation = None
        else:
            raise ValueError(f"不支持的调制方式: {mod_type}")

        # 构建星座参考
        self._constellation_keys = sorted(self._map.keys())
        self._constellation_vals = np.array([self._map[k] for k in self._constellation_keys])

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """PSK解调。

        :param signal: 复基带IQ信号

        :returns: 解调后的比特序列
        """
        # 匹配滤波
        filtered = np.convolve(signal, self._rrc, mode='same')

        # 降采样（取峰值点）
        symbols = filtered[::self.sps]

        # 反旋转 + 最近邻判决
        bps = self._bits_per_symbol
        bits = []
        for i, sym in enumerate(symbols):
            # 反相位旋转
            if self._rotation is not None and (i % 2 == 1):
                sym = sym * np.conj(self._rotation)

            # 最近邻判决
            distances = np.abs(sym - self._constellation_vals)
            idx = self._constellation_keys[np.argmin(distances)]

            # 解码比特
            bits.extend((idx >> b) & 1 for b in range(bps - 1, -1, -1))

        return np.array(bits, dtype=int)
