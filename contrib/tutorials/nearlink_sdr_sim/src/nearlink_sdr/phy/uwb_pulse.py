"""超宽带脉冲波形与调制 (标准 6.2.1.4)。

实现 Kaiser 参考脉冲波形、码片载波调制和归一化互相关验证。
"""

from __future__ import annotations

__all__ = [
    "UWBPulseConfig",
    "chip_modulate",
    "kaiser_pulse",
    "normalized_cross_correlation",
    "validate_pulse",
]


import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _i0(x: float) -> float:
    """第一类零阶修正贝塞尔函数。"""
    return float(np.i0(x))


@dataclass
class UWBPulseConfig:
    """UWB 脉冲配置参数。

    :ivar tp_ns: 带宽参数 Tp (ns)
    :ivar tw_ns: 归一化互相关函数参数 Tw (ns)
    :ivar beta: Kaiser 波形形状参数
    :ivar max_prf_mhz: 最大脉冲重复频率 (MHz)
    :ivar sample_rate_ghz: 采样率 (GHz)
    """
    tp_ns: float = 2.0
    tw_ns: float = 0.5
    beta: float = 10.0
    max_prf_mhz: float = 499.2
    sample_rate_ghz: float = 4.0

    @classmethod
    def bw_500mhz(cls) -> UWBPulseConfig:
        """创建 -10dB 带宽 >= 500MHz 的配置。"""
        return cls(tp_ns=2.0, tw_ns=0.5)

    @classmethod
    def bw_1300mhz(cls) -> UWBPulseConfig:
        """创建 -10dB 带宽 > 1300MHz 的配置。"""
        return cls(tp_ns=0.75, tw_ns=0.2)

    @property
    def pulse_duration_ns(self) -> float:
        """脉冲持续时间 L = 3 * Tp (ns)。"""
        return 3.0 * self.tp_ns

    @property
    def chip_duration_ns(self) -> float:
        """码片持续时间 Tc = 1 / max_prf (ns)。"""
        return 1e3 / self.max_prf_mhz

    @property
    def samples_per_chip(self) -> int:
        """每个码片的采样点数。"""
        return round(self.chip_duration_ns * self.sample_rate_ghz)


def kaiser_pulse(
    cfg: UWBPulseConfig,
    num_samples: int | None = None,
) -> NDArray[np.float64]:
    """生成 Kaiser 参考脉冲波形 r(t)。

    :param cfg: UWB 脉冲配置
    :param num_samples: 采样点数, 默认根据脉冲持续时间和采样率计算

    :returns: 归一化的 Kaiser 脉冲波形采样值
    """
    l_ns = cfg.pulse_duration_ns
    if num_samples is None:
        num_samples = math.ceil(l_ns * cfg.sample_rate_ghz) + 1

    t = np.linspace(-l_ns / 2, l_ns / 2, num_samples)
    i0_beta = _i0(cfg.beta)
    r = np.zeros_like(t)

    mask = np.abs(t) <= l_ns / 2
    arg = cfg.beta * np.sqrt(np.maximum(1.0 - (2.0 * t[mask] / l_ns) ** 2, 0.0))
    r[mask] = np.i0(arg) / (l_ns * i0_beta)

    return r


def chip_modulate(
    chips: NDArray[np.int8],
    cfg: UWBPulseConfig,
    fc_ghz: float,
) -> NDArray[np.float64]:
    """码片载波调制。

    每个码片 c(n) ∈ {-1, 0, +1} 的调制信号为:
        c(n) · r(t - n·Tc) · cos[2π·fc·(t - n·Tc)]

    :param chips: 码片序列, 值为 -1, 0 或 +1
    :param cfg: UWB 脉冲配置
    :param fc_ghz: 载波中心频率 (GHz)

    :returns: 调制后的时域信号
    """
    pulse = kaiser_pulse(cfg)
    pulse_len = len(pulse)
    spc = cfg.samples_per_chip
    total_len = spc * len(chips) + pulse_len - 1

    signal = np.zeros(total_len, dtype=np.float64)
    dt_ns = 1.0 / cfg.sample_rate_ghz

    for n, c in enumerate(chips):
        if c == 0:
            continue
        start = n * spc
        t_offset_ns = n * cfg.chip_duration_ns
        t_local = np.arange(pulse_len) * dt_ns
        carrier = np.cos(2.0 * np.pi * fc_ghz * (t_local + start * dt_ns - t_offset_ns))
        signal[start:start + pulse_len] += float(c) * pulse * carrier

    return signal


def normalized_cross_correlation(
    r: NDArray[np.float64],
    p: NDArray[np.float64],
) -> NDArray[np.float64]:
    """计算归一化互相关函数 |φ(τ)|。

    :param r: 参考脉冲波形
    :param p: 发射脉冲波形

    :returns: 归一化互相关函数的幅度
    """
    e_r = np.sum(r ** 2)
    e_p = np.sum(p ** 2)
    if e_r == 0 or e_p == 0:
        return np.zeros(len(r) + len(p) - 1)

    corr = np.correlate(r, p, mode="full")
    norm = math.sqrt(e_r * e_p)
    return np.abs(corr / norm)


def validate_pulse(
    p: NDArray[np.float64],
    cfg: UWBPulseConfig,
    main_lobe_threshold: float = 0.92,
    side_lobe_threshold: float = 0.1,
) -> tuple[bool, float, float]:
    """验证发射脉冲波形是否满足标准要求。

    :param p: 发射脉冲波形
    :param cfg: UWB 脉冲配置
    :param main_lobe_threshold: 主瓣最低强度阈值 (默认 0.92)
    :param side_lobe_threshold: 旁瓣最高强度阈值 (默认 0.1)

    :returns: (是否通过, 主瓣最小强度, 旁瓣最大强度)
    """
    r = kaiser_pulse(cfg, num_samples=len(p))
    phi = normalized_cross_correlation(r, p)

    peak_idx = np.argmax(phi)
    peak_val = phi[peak_idx]

    # 主瓣: 围绕峰值, 持续时间 >= Tw 的区域
    tw_samples = math.ceil(cfg.tw_ns * cfg.sample_rate_ghz)
    half_tw = tw_samples // 2
    start = max(0, peak_idx - half_tw)
    end = min(len(phi), peak_idx + half_tw + 1)
    main_lobe_min = float(np.min(phi[start:end]))

    # 旁瓣: 标准定义为 d|φ(τ)|/dτ = 0 的临界点 (局部极值)
    # 先找主瓣的连续区域 (从峰值向两侧单调递减直到第一个局部极小值)
    left_boundary = peak_idx
    while left_boundary > 0 and phi[left_boundary - 1] <= phi[left_boundary]:
        left_boundary -= 1
    right_boundary = peak_idx
    while right_boundary < len(phi) - 1 and phi[right_boundary + 1] <= phi[right_boundary]:
        right_boundary += 1

    # 在主瓣以外查找局部极大值
    side_lobe_max = 0.0
    for region in (phi[:left_boundary], phi[right_boundary + 1:]):
        if len(region) < 3:
            continue
        for i in range(1, len(region) - 1):
            if region[i] >= region[i - 1] and region[i] >= region[i + 1]:
                side_lobe_max = max(side_lobe_max, float(region[i]))
    # 边界值也作为候选
    if left_boundary > 0:
        side_lobe_max = max(side_lobe_max, float(phi[0]))
    if right_boundary < len(phi) - 1:
        side_lobe_max = max(side_lobe_max, float(phi[-1]))

    main_ok = main_lobe_min >= main_lobe_threshold
    side_ok = side_lobe_max <= side_lobe_threshold
    _ = peak_val  # 避免未使用警告

    return (main_ok and side_ok, main_lobe_min, side_lobe_max)
