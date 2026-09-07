"""多音信号生成器 -- TXS-10002-2025 标准 6.2.1.3

基带多音信号: S(t) = sum_{i=1}^{N} A_i * exp(j * (omega_i * t + phi_i))

标准规定 N = 1, 2, 4, 8, 各音等幅、等间隔、对称分布。
测量信号 2 (6.2.4.2) 引用本模块生成多音波形。
"""

from __future__ import annotations

__all__ = [
    "MultitoneConfig",
    "generate_multitone",
    "multitone_peak_to_avg_ratio",
]


from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 6.2.4.2 多音信号参数表
# ---------------------------------------------------------------------------

# 初始相位集合 (弧度), 按 N 和 set 索引
# N=2: 固定 phi=[0, 0]
# N=4: Set 1 / Set 2
# N=8: Set 1 / Set 2

MULTITONE_PHASE_SETS: dict[int, dict[int, list[float]]] = {
    2: {
        1: [0.0, 0.0],
    },
    4: {
        1: [1.81, 0.0, 0.0, 1.81],
        2: [4.47, 0.0, 0.0, 4.47],
    },
    8: {
        1: [5.40, 3.84, 0.91, 0.0, 0.0, 0.91, 3.84, 5.40],
        2: [0.88, 2.45, 5.37, 0.0, 0.0, 5.37, 2.45, 0.88],
    },
}

# SLE 带宽 (MHz) -> N -> 相邻音频率间隔 delta_f (MHz)
DELTA_F_TABLE: dict[int, dict[int, float]] = {
    1: {2: 0.5, 4: 0.25, 8: 0.125},
    2: {2: 1.0, 4: 0.5, 8: 0.25},
    4: {2: 2.0, 4: 1.0, 8: 0.5},
}


@dataclass
class MultitoneConfig:
    """多音信号配置。"""
    n_tones: int = 1              # 音的个数 (1/2/4/8)
    sle_bandwidth_mhz: int = 1    # SLE 带宽 (1/2/4 MHz)
    phase_set: int = 1            # 相位集合 (1 或 2, N>=4 时有效)
    amplitude: float = 1.0        # 各音等幅
    sample_rate_hz: float = 8e6   # 采样率 (Hz)

    @property
    def delta_f_hz(self) -> float:
        """相邻音频率间隔 (Hz)。"""
        if self.n_tones == 1:
            return 0.0
        return DELTA_F_TABLE[self.sle_bandwidth_mhz][self.n_tones] * 1e6

    @property
    def frequencies_hz(self) -> list[float]:
        """各音的基带频率列表 (Hz), 中心对称分布。"""
        if self.n_tones == 1:
            return [0.0]
        df = self.delta_f_hz
        n = self.n_tones
        # 对称分布: -(N/2-0.5)*df, ..., -0.5*df, +0.5*df, ..., +(N/2-0.5)*df
        return [(-n / 2 + i + 0.5) * df for i in range(n)]

    @property
    def phases_rad(self) -> list[float]:
        """各音的初始相位 (rad)。"""
        if self.n_tones == 1:
            return [0.0]
        return MULTITONE_PHASE_SETS[self.n_tones][self.phase_set]


def generate_multitone(
    config: MultitoneConfig,
    duration_s: float,
) -> np.ndarray:
    """生成基带多音信号 (复数采样)。

    :param config: 多音信号配置.
    :param duration_s: 信号持续时间 (秒).

    :returns: 复数基带采样序列, shape=(num_samples,).
    """
    num_samples = int(config.sample_rate_hz * duration_s)
    t = np.arange(num_samples) / config.sample_rate_hz
    freqs = config.frequencies_hz
    phases = config.phases_rad
    signal = np.zeros(num_samples, dtype=np.complex128)
    for i in range(config.n_tones):
        signal += config.amplitude * np.exp(
            1j * (2 * np.pi * freqs[i] * t + phases[i])
        )
    return signal


def multitone_peak_to_avg_ratio(config: MultitoneConfig) -> float:
    """计算多音信号的峰均比 (线性)。

    对于等幅多音, 理论最大 PAPR = N (所有音同相叠加时).
    """
    return float(config.n_tones)
