"""射频合规参数与校验 — 标准第 8 章。

实现 TXS-10002-2025 标准中 8.2-8.4 定义的窄带/UWB 射频指标:
  - 8.2.1  输出功率等级
  - 8.2.2  调制精度 (GFSK 频偏 / PSK EVM / 频率容限 / 时钟精度)
  - 8.2.3  无用发射 (GFSK/PSK 频段内杂散)
  - 8.3.1  接收灵敏度与最大输入电平
  - 8.3.2  接收机选择性
  - 8.3.4  RSSI 精度
  - 8.4    UWB 射频 (信道、频谱模板、频率容限、均方根误差)
"""

from __future__ import annotations

__all__ = [
    "ACTIVE_CLOCK_JITTER_US",
    "ACTIVE_CLOCK_PPM",
    "MAX_INPUT_LEVEL_DBM",
    "RSSI_ACCURACY_DB",
    "RSSI_SIGNAL_OFFSET_DB",
    "SLEEP_CLOCK_JITTER_US",
    "SLEEP_CLOCK_PPM",
    "SUB_1G_TOLERANCE_PPM",
    "UWB_CLOCK_TOLERANCE_PPM",
    "UWB_FREQ_TOLERANCE_PPM",
    "UWB_MAX_NRMSE_DIFF_DB",
    "UWB_MAX_NRMSE_PCT",
    "ChannelBandwidth",
    "EVMLimit",
    "FreqBand",
    "FreqToleranceSpec",
    "GFSKFreqDevResult",
    "GFSKFreqDevSpec",
    "IntermodEntry",
    "OutOfBandEntry",
    "PSKModulation",
    "PSKSpectrumMask",
    "PowerClass",
    "RFComplianceReport",
    "SelectivityEntry",
    "UWBSpectrumMask",
    "check_clock_accuracy",
    "check_evm",
    "check_freq_tolerance",
    "check_gfsk_freq_dev",
    "check_rssi_accuracy",
    "check_rx_spurious_emission",
    "check_uwb_nrmse",
    "classify_power",
    "get_psk_spectrum_mask",
    "gfsk_inband_spurious_limit_dbm",
    "reference_sensitivity",
    "selectivity_freq_offsets",
    "uwb_channel_center_freq_mhz",
    "validate_power_step",
]


from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


# ---------------------------------------------------------------------------
# 8.2.1 输出功率等级 (表 52)
# ---------------------------------------------------------------------------
class PowerClass(IntEnum):
    CLASS_0 = 0   # Pmax > 20 dBm
    CLASS_1 = 1   # 17 < Pmax <= 20 dBm
    CLASS_2 = 2   # 14 < Pmax <= 17 dBm
    CLASS_3 = 3   # 10 < Pmax <= 14 dBm
    CLASS_4 = 4   # 4 < Pmax <= 10 dBm
    CLASS_5 = 5   # 0 < Pmax <= 4 dBm
    CLASS_6 = 6   # Pmax <= 0 dBm


_POWER_CLASS_UPPER = {
    PowerClass.CLASS_0: float('inf'),
    PowerClass.CLASS_1: 20.0,
    PowerClass.CLASS_2: 17.0,
    PowerClass.CLASS_3: 14.0,
    PowerClass.CLASS_4: 10.0,
    PowerClass.CLASS_5: 4.0,
    PowerClass.CLASS_6: 0.0,
}

_POWER_CLASS_LOWER = {
    PowerClass.CLASS_0: 20.0,
    PowerClass.CLASS_1: 17.0,
    PowerClass.CLASS_2: 14.0,
    PowerClass.CLASS_3: 10.0,
    PowerClass.CLASS_4: 4.0,
    PowerClass.CLASS_5: 0.0,
    PowerClass.CLASS_6: float('-inf'),
}


def classify_power(pmax_dbm: float) -> PowerClass:
    """根据最大输出功率确定功率等级。"""
    for cls in PowerClass:
        lo = _POWER_CLASS_LOWER[cls]
        hi = _POWER_CLASS_UPPER[cls]
        if lo < pmax_dbm <= hi:
            return cls
    if pmax_dbm > 20.0:
        return PowerClass.CLASS_0
    return PowerClass.CLASS_6


def validate_power_step(prev_dbm: float, curr_dbm: float) -> bool:
    """校验相邻发射功率差是否不大于 8 dB (8.2.1)。"""
    return abs(curr_dbm - prev_dbm) <= 8.0


# ---------------------------------------------------------------------------
# 8.2.2.1 GFSK 频率偏差 (表 + 文字)
# ---------------------------------------------------------------------------
@dataclass
class GFSKFreqDevSpec:
    """GFSK 调制频偏规范 — 针对单个符号速率。"""
    symbol_rate_msps: float
    symbol_interval_us: float
    min_deviation_khz: float       # 任意序列最小频偏
    fd1_min_khz: float             # 11110000 序列 fd1 下限
    fd1_max_khz: float             # 11110000 序列 fd1 上限
    fd2_min_khz: float             # 10101010 序列 fd2 下限
    fd2_fd1_ratio_min: float = 0.8  # fd2/fd1 均值比最小值
    zero_crossing_margin: float = 0.125  # 过零误差 < ±1/8 符号周期


# 标准表中 7 种符号速率的规范
GFSK_FREQ_DEV_SPECS: dict[float, GFSKFreqDevSpec] = {
    0.1: GFSKFreqDevSpec(
        0.1, 10.0, 18.5, 22.5, 27.5, 18.5),
    0.125: GFSKFreqDevSpec(
        0.125, 8.0, 23.125, 28.125, 34.375, 23.125),
    0.25: GFSKFreqDevSpec(
        0.25, 4.0, 46.25, 56.25, 68.75, 46.25),
    0.5: GFSKFreqDevSpec(
        0.5, 2.0, 92.5, 112.5, 137.5, 92.5),
    1.0: GFSKFreqDevSpec(
        1.0, 1.0, 185.0, 225.0, 275.0, 185.0),
    2.0: GFSKFreqDevSpec(
        2.0, 0.5, 370.0, 450.0, 550.0, 370.0),
    4.0: GFSKFreqDevSpec(
        4.0, 0.25, 740.0, 900.0, 1100.0, 740.0),
}


@dataclass
class GFSKFreqDevResult:
    """GFSK 频偏校验结果。"""
    min_ok: bool            # 任意序列最小频偏是否达标
    fd1_ok: bool            # fd1 是否在 [fd1_min, fd1_max]
    fd2_ok: bool            # fd2 是否 >= fd2_min
    ratio_ok: bool          # fd2/fd1 >= 0.8
    zero_crossing_ok: bool  # 过零误差是否合规

    @property
    def passed(self) -> bool:
        return all([self.min_ok, self.fd1_ok, self.fd2_ok,
                    self.ratio_ok, self.zero_crossing_ok])


def check_gfsk_freq_dev(
    symbol_rate_msps: float,
    min_deviation_khz: float,
    fd1_khz: float,
    fd2_khz: float,
    max_zero_crossing_error: float = 0.0,
) -> GFSKFreqDevResult:
    """校验 GFSK 频率偏差是否满足标准 8.2.2.1。"""
    spec = GFSK_FREQ_DEV_SPECS.get(symbol_rate_msps)
    if spec is None:
        raise ValueError(f"不支持的符号速率: {symbol_rate_msps} Msps")

    return GFSKFreqDevResult(
        min_ok=min_deviation_khz >= spec.min_deviation_khz,
        fd1_ok=spec.fd1_min_khz <= fd1_khz <= spec.fd1_max_khz,
        fd2_ok=fd2_khz >= spec.fd2_min_khz,
        ratio_ok=(fd2_khz / fd1_khz >= spec.fd2_fd1_ratio_min
                  if fd1_khz > 0 else False),
        zero_crossing_ok=max_zero_crossing_error <= spec.zero_crossing_margin,
    )


# ---------------------------------------------------------------------------
# 8.2.2.2 PSK EVM (表 53)
# ---------------------------------------------------------------------------
class PSKModulation(IntEnum):
    PI2_BPSK = 0
    PI4_QPSK = 1
    PI8_8PSK = 2


@dataclass
class EVMLimit:
    """EVM 限值。"""
    rms_pct: float
    pct99: float
    peak_pct: float


PSK_EVM_LIMITS: dict[PSKModulation, EVMLimit] = {
    PSKModulation.PI2_BPSK: EVMLimit(20.0, 40.0, 45.0),
    PSKModulation.PI4_QPSK: EVMLimit(13.0, 28.0, 32.0),
    PSKModulation.PI8_8PSK: EVMLimit(9.0, 20.0, 25.0),
}


def check_evm(
    mod: PSKModulation,
    rms_evm_pct: float,
    pct99_evm_pct: float,
    peak_evm_pct: float,
) -> bool:
    """校验 PSK EVM 是否满足表 53 限值。"""
    lim = PSK_EVM_LIMITS[mod]
    return (rms_evm_pct <= lim.rms_pct
            and pct99_evm_pct <= lim.pct99
            and peak_evm_pct <= lim.peak_pct)


# ---------------------------------------------------------------------------
# 8.2.2.3 频率容限 (表 54/55)
# ---------------------------------------------------------------------------
class FreqBand(IntEnum):
    BAND_2400 = 0
    BAND_5100 = 1
    BAND_5800 = 2
    SUB_1G = 3


@dataclass
class FreqToleranceSpec:
    """频率容限规范。"""
    freq_offset_khz: float
    freq_drift_khz: float
    drift_rate_hz_per_us: float


# 表 54: 2400 MHz
FREQ_TOLERANCE_2400: dict[str, FreqToleranceSpec] = {
    "GFSK": FreqToleranceSpec(150.0, 50.0, 400.0),
    "BPSK/QPSK": FreqToleranceSpec(48.0, 10.0, 34.0),
    "8PSK": FreqToleranceSpec(48.0, 10.0, 11.0),
}

# 表 55: 5100/5800 MHz
FREQ_TOLERANCE_5G: dict[str, FreqToleranceSpec] = {
    "GFSK": FreqToleranceSpec(175.0, 100.0, 400.0),
    "BPSK/QPSK": FreqToleranceSpec(120.0, 20.0, 34.0),
    "8PSK": FreqToleranceSpec(120.0, 20.0, 11.0),
}

# sub-1GHz: ±100 ppm
SUB_1G_TOLERANCE_PPM = 100.0


def check_freq_tolerance(
    band: FreqBand,
    mod_type: str,
    freq_offset_khz: float,
    freq_drift_khz: float,
    drift_rate_hz_per_us: float,
    carrier_freq_mhz: float = 0.0,
) -> bool:
    """校验频率容限是否满足 8.2.2.3。

    :param band: 频段类型。
    :param mod_type: 调制类型 ("GFSK", "BPSK/QPSK", "8PSK")。
    :param freq_offset_khz: 初始频率偏差 (kHz)。
    :param freq_drift_khz: 频率漂移 (kHz)。
    :param drift_rate_hz_per_us: 频率漂移率 (Hz/us)。
    :param carrier_freq_mhz: 载波频率 (MHz), sub-1GHz 时用于 ppm 校验。
    """
    if band == FreqBand.SUB_1G:
        if carrier_freq_mhz <= 0:
            return False
        max_offset_khz = carrier_freq_mhz * SUB_1G_TOLERANCE_PPM * 1e-3
        return abs(freq_offset_khz) <= max_offset_khz

    table = (FREQ_TOLERANCE_2400
             if band == FreqBand.BAND_2400
             else FREQ_TOLERANCE_5G)
    spec = table.get(mod_type)
    if spec is None:
        raise ValueError(f"不支持的调制类型: {mod_type}")

    return (abs(freq_offset_khz) <= spec.freq_offset_khz
            and abs(freq_drift_khz) <= spec.freq_drift_khz
            and abs(drift_rate_hz_per_us) <= spec.drift_rate_hz_per_us)


# ---------------------------------------------------------------------------
# 8.2.2.4 / 8.2.2.5 时钟精度
# ---------------------------------------------------------------------------
ACTIVE_CLOCK_PPM = 20.0
ACTIVE_CLOCK_JITTER_US = 2.0
SLEEP_CLOCK_PPM = 500.0
SLEEP_CLOCK_JITTER_US = 16.0


def check_clock_accuracy(
    ppm: float,
    jitter_us: float,
    is_sleep: bool = False,
) -> bool:
    """校验时钟精度 (8.2.2.4 / 8.2.2.5)。"""
    max_ppm = SLEEP_CLOCK_PPM if is_sleep else ACTIVE_CLOCK_PPM
    max_jitter = SLEEP_CLOCK_JITTER_US if is_sleep else ACTIVE_CLOCK_JITTER_US
    return ppm <= max_ppm and jitter_us <= max_jitter


# ---------------------------------------------------------------------------
# 8.2.3 无用发射 — GFSK 频段内杂散 (表 56/57)
# ---------------------------------------------------------------------------
class ChannelBandwidth(IntEnum):
    BW_100K = 100
    BW_125K = 125
    BW_250K = 250
    BW_500K = 500
    BW_1M = 1000
    BW_2M = 2000
    BW_4M = 4000


def gfsk_inband_spurious_limit_dbm(
    bw: ChannelBandwidth,
    k_offset_mhz: float,
) -> float:
    """计算 GFSK 频段内杂散限值 (dBm)。

    对宽信道 (1/2/4 MHz, 表56) 返回 dBm/MHz;
    对窄信道 (100-500 kHz, 表57) 返回 dBm/100kHz。
    """
    abs_k = abs(k_offset_mhz)

    if bw == ChannelBandwidth.BW_1M:
        if abs_k == 2:
            return -20.0
        return -30.0
    if bw == ChannelBandwidth.BW_2M:
        if abs_k == 4:
            return -20.0
        if abs_k == 5:
            return -23.0
        return -30.0
    if bw == ChannelBandwidth.BW_4M:
        if abs_k in (7, 8, 9):
            return -20.0
        if abs_k == 10:
            return -23.0
        return -30.0

    # 窄信道 (100/125/250/500 kHz, 表57)
    bw_mhz = bw / 1000.0
    first_adj = {100: 0.2, 125: 0.25, 250: 0.5, 500: 1.0}
    f1 = first_adj.get(int(bw), bw_mhz)

    if abs_k <= f1:
        return -30.0
    return -40.0


# ---------------------------------------------------------------------------
# 8.2.3.2 PSK 频谱模板 (表 58/59)
# ---------------------------------------------------------------------------
@dataclass
class PSKSpectrumMask:
    """PSK 发射频谱模板中各参考偏移频率 (MHz)。"""
    bw: ChannelBandwidth
    f1_mhz: float
    f2_mhz: float
    f3_mhz: float | None = None  # 窄信道仅有 f1/f2/f3
    f4_mhz: float | None = None  # 宽信道有 f4


# 表 58 — 宽信道
PSK_SPECTRUM_MASK_WIDE: dict[ChannelBandwidth, PSKSpectrumMask] = {
    ChannelBandwidth.BW_1M: PSKSpectrumMask(
        ChannelBandwidth.BW_1M, 1.0, 1.5, None, 2.5),
    ChannelBandwidth.BW_2M: PSKSpectrumMask(
        ChannelBandwidth.BW_2M, 2.0, 3.5, 4.5, 5.5),
    ChannelBandwidth.BW_4M: PSKSpectrumMask(
        ChannelBandwidth.BW_4M, 4.0, 6.5, 9.5, 10.5),
}

# 表 59 — 窄信道
PSK_SPECTRUM_MASK_NARROW: dict[ChannelBandwidth, PSKSpectrumMask] = {
    ChannelBandwidth.BW_100K: PSKSpectrumMask(
        ChannelBandwidth.BW_100K, 0.1, 0.15, 0.25),
    ChannelBandwidth.BW_125K: PSKSpectrumMask(
        ChannelBandwidth.BW_125K, 0.125, 0.1875, 0.3125),
    ChannelBandwidth.BW_250K: PSKSpectrumMask(
        ChannelBandwidth.BW_250K, 0.25, 0.375, 0.625),
    ChannelBandwidth.BW_500K: PSKSpectrumMask(
        ChannelBandwidth.BW_500K, 0.5, 0.75, 1.25),
}


def get_psk_spectrum_mask(bw: ChannelBandwidth) -> PSKSpectrumMask:
    """获取指定带宽的 PSK 频谱模板。"""
    if bw in PSK_SPECTRUM_MASK_WIDE:
        return PSK_SPECTRUM_MASK_WIDE[bw]
    if bw in PSK_SPECTRUM_MASK_NARROW:
        return PSK_SPECTRUM_MASK_NARROW[bw]
    raise ValueError(f"不支持的信道带宽: {bw}")


# ---------------------------------------------------------------------------
# 8.3.1.1 参考灵敏度 (表 60/61)
# ---------------------------------------------------------------------------
_BW_KEYS = [100, 125, 250, 500, 1000, 2000, 4000]

# 表 60 — GFSK 灵敏度 (dBm)
GFSK_SENSITIVITY: dict[int, float] = {
    100: -79.9, 125: -79.0, 250: -76.0, 500: -73.0,
    1000: -70.0, 2000: -67.0, 4000: -64.0,
}

# 表 61 — PSK 灵敏度 (dBm), 按 (MCS, bw_khz) 索引
PSK_SENSITIVITY: dict[tuple[int, int], float] = {}
_PSK_ROWS = [
    (0, [-90.9, -90.0, -87.0, -84.0, -81.0, -78.0, -75.0]),
    (1, [-88.9, -88.0, -85.0, -82.0, -79.0, -76.0, -73.0]),
    (2, [-87.9, -87.0, -84.0, -81.0, -78.0, -75.0, -72.0]),
    (3, [-85.9, -85.0, -82.0, -79.0, -76.0, -73.0, -70.0]),
    (4, [-84.9, -84.0, -81.0, -77.0, -75.0, -72.0, -69.0]),
    (5, [-82.9, -82.0, -79.0, -75.0, -73.0, -70.0, -67.0]),
    (6, [-81.9, -81.0, -78.0, -74.0, -72.0, -69.0, -66.0]),
    (7, [-79.9, -79.0, -76.0, -72.0, -70.0, -67.0, -64.0]),
    (8, [-75.9, -75.0, -72.0, -68.0, -66.0, -63.0, -60.0]),
    (9, [-78.9, -78.0, -75.0, -71.0, -69.0, -66.0, -63.0]),
    (10, [-77.9, -77.0, -74.0, -70.0, -68.0, -65.0, -62.0]),
    (11, [-75.9, -75.0, -72.0, -68.0, -66.0, -63.0, -60.0]),
    (12, [-69.9, -69.0, -66.0, -62.0, -60.0, -57.0, -54.0]),
]
for _mcs, _vals in _PSK_ROWS:
    for _bw, _v in zip(_BW_KEYS, _vals, strict=True):
        PSK_SENSITIVITY[(_mcs, _bw)] = _v


def reference_sensitivity(
    bw_khz: int,
    mcs: int | None = None,
) -> float:
    """返回参考灵敏度 (dBm)。

    :param bw_khz: 信道带宽 (kHz)。
    :param mcs: MCS 索引。None 时返回 GFSK 灵敏度。
    """
    if mcs is None:
        val = GFSK_SENSITIVITY.get(bw_khz)
        if val is None:
            raise ValueError(f"不支持的 GFSK 带宽: {bw_khz}kHz")
        return val
    key = (mcs, bw_khz)
    val = PSK_SENSITIVITY.get(key)
    if val is None:
        raise ValueError(f"不支持的 PSK 灵敏度查询: MCS={mcs}, BW={bw_khz}kHz")
    return val


# 8.3.1.2 最大输入电平
MAX_INPUT_LEVEL_DBM = -10.0


# ---------------------------------------------------------------------------
# 8.3.2.1 接收机选择性 (表 62/63)
# ---------------------------------------------------------------------------
@dataclass
class SelectivityEntry:
    """接收机选择性要求条目。"""
    signal_level_offset_db: float  # 相对于参考灵敏度的偏移
    interferer_level_dbm: float
    freq_offset_type: str  # "0", "f1", "f2", "f3+", "image", "image_f1"


# 表 62
SELECTIVITY_TABLE: list[SelectivityEntry] = [
    SelectivityEntry(3.0, -88.0, "0"),
    SelectivityEntry(3.0, -82.0, "f1"),
    SelectivityEntry(3.0, -50.0, "f2"),
    SelectivityEntry(3.0, -40.0, "f3+"),
    SelectivityEntry(3.0, -58.0, "image"),
    SelectivityEntry(3.0, -52.0, "image_f1"),
]

# 表 63 — 干扰信号频率偏移 (MHz)
SELECTIVITY_FREQ_OFFSET: dict[int, tuple[float, float, float]] = {
    100: (0.1, 0.2, 0.3),
    125: (0.125, 0.25, 0.375),
    250: (0.25, 0.5, 0.75),
    500: (0.5, 1.0, 1.5),
    1000: (1.0, 2.0, 3.0),
    2000: (2.0, 4.0, 6.0),
    4000: (4.0, 8.0, 12.0),
}


def selectivity_freq_offsets(
    bw_khz: int,
) -> tuple[float, float, float]:
    """返回指定带宽的 f1/f2/f3 频率偏移 (MHz)。"""
    val = SELECTIVITY_FREQ_OFFSET.get(bw_khz)
    if val is None:
        raise ValueError(f"不支持的选择性带宽: {bw_khz}kHz")
    return val


# ---------------------------------------------------------------------------
# 8.3.2.2 频段外选择性 (表 64/65/66)
# ---------------------------------------------------------------------------
@dataclass
class OutOfBandEntry:
    """频段外选择性要求条目。"""
    freq_range: tuple[float, float]  # (start_mhz, end_mhz)
    interferer_level_dbm: float
    step_mhz: float


OOB_SUB1G: list[OutOfBandEntry] = [
    OutOfBandEntry((30.0, 313.0), -35.0, 1.0),
    OutOfBandEntry((929.0, 3000.0), -35.0, 1.0),
    OutOfBandEntry((3000.0, 12750.0), -30.0, 2.5),
]

OOB_2400: list[OutOfBandEntry] = [
    OutOfBandEntry((30.0, 2000.0), -30.0, 10.0),
    OutOfBandEntry((2003.0, 2399.0), -35.0, 3.0),
    OutOfBandEntry((2484.0, 2997.0), -35.0, 3.0),
    OutOfBandEntry((3000.0, 12750.0), -30.0, 25.0),
]

OOB_5G: list[OutOfBandEntry] = [
    OutOfBandEntry((30.0, 4500.0), -30.0, 10.0),
    OutOfBandEntry((4510.0, 5130.0), -35.0, 10.0),
    OutOfBandEntry((5870.0, 6490.0), -35.0, 10.0),
    OutOfBandEntry((6500.0, 12750.0), -30.0, 25.0),
]


# ---------------------------------------------------------------------------
# 8.3.2.3 干扰互调 (表 67)
# ---------------------------------------------------------------------------
@dataclass
class IntermodEntry:
    """接收机干扰互调要求。"""
    bw_khz: int
    signal_offset_db: float            # 有用信号相对灵敏度偏移
    interferer_level_dbm: float
    n_values: tuple[int, ...]          # |f2-f1| = n * bw_mhz


INTERMOD_TABLE: list[IntermodEntry] = [
    IntermodEntry(bw, 6.0, -50.0, (3, 4, 5))
    for bw in [100, 125, 250, 500, 1000, 2000, 4000]
]


# ---------------------------------------------------------------------------
# 8.3.4 RSSI 测量精度
# ---------------------------------------------------------------------------
RSSI_ACCURACY_DB = 6.0
RSSI_SIGNAL_OFFSET_DB = 6.0


def check_rssi_accuracy(
    measured_rssi_dbm: float,
    actual_level_dbm: float,
    ref_sensitivity_dbm: float,
) -> bool:
    """校验 RSSI 测量精度 (8.3.4)。

    在参考灵敏度 +6 dB 处, 误差应不大于 ±6 dB。
    """
    expected_level = ref_sensitivity_dbm + RSSI_SIGNAL_OFFSET_DB
    if actual_level_dbm < expected_level - 0.5:
        return True  # 不在测试条件范围内
    return abs(measured_rssi_dbm - actual_level_dbm) <= RSSI_ACCURACY_DB


# ---------------------------------------------------------------------------
# 8.4 UWB 射频要求
# ---------------------------------------------------------------------------
def uwb_channel_center_freq_mhz(nc: int) -> float:
    """计算 UWB 物理信道中心频率 (MHz)。

    fc = 499.2 + Nc * 124.8  (Nc in 0..79)
    特殊信道: 125 -> 7542.6, 126 -> 18041.8, 127 -> 8541.0
    """
    special = {125: 7542.6, 126: 18041.8, 127: 8541.0}
    if nc in special:
        return special[nc]
    if 0 <= nc <= 79:
        return 499.2 + nc * 124.8
    raise ValueError(f"UWB 物理信道号超出范围: {nc}")


@dataclass
class UWBSpectrumMask:
    """UWB 频谱模板参数 (8.4.3)。"""
    pulse_width_ns: float  # Tp

    def transition_band_hz(self) -> tuple[float, float]:
        """过渡带范围 [0.65/Tp, 0.8/Tp] Hz。"""
        tp_s = self.pulse_width_ns * 1e-9
        return (0.65 / tp_s, 0.8 / tp_s)

    def check_spectrum(
        self,
        freq_offset_hz: float,
        psd_db_relative: float,
    ) -> bool:
        """校验 PSD 是否满足频谱模板要求。

        - |f-fc| in [0.65/Tp, 0.8/Tp]: 低于峰值 10 dB
        - |f-fc| > 0.8/Tp: 低于峰值 18 dB
        """
        tp_s = self.pulse_width_ns * 1e-9
        abs_offset = abs(freq_offset_hz)
        boundary_lo = 0.65 / tp_s
        boundary_hi = 0.8 / tp_s

        if abs_offset < boundary_lo:
            return True
        if abs_offset <= boundary_hi:
            return psd_db_relative <= -10.0
        return psd_db_relative <= -18.0


# 8.4.4 UWB 频率容限与时钟偏差
UWB_FREQ_TOLERANCE_PPM = 20.0
UWB_CLOCK_TOLERANCE_PPM = 20.0

# 8.4.5 UWB 均方根误差
UWB_MAX_NRMSE_PCT = 25.0
UWB_MAX_NRMSE_DIFF_DB = 2.0


def check_uwb_nrmse(
    sync_nrmse_pct: float,
    meas_nrmse_pct: float,
) -> bool:
    """校验 UWB 标准化均方根误差 (8.4.5)。

    同步信号和测量信号 NRMSE 均不超过 25%,
    且两者差值不超过 ±2 dB。
    """
    if sync_nrmse_pct > UWB_MAX_NRMSE_PCT:
        return False
    if meas_nrmse_pct > UWB_MAX_NRMSE_PCT:
        return False
    # 差值以 dB 表示: 20*log10(ratio)
    if sync_nrmse_pct <= 0 or meas_nrmse_pct <= 0:
        return False
    ratio_db = 20.0 * np.log10(sync_nrmse_pct / meas_nrmse_pct)
    return abs(ratio_db) <= UWB_MAX_NRMSE_DIFF_DB


# ---------------------------------------------------------------------------
# 8.3.3 接收机杂散发射
# ---------------------------------------------------------------------------
def check_rx_spurious_emission(
    spurious_dbm: float,
    limit_dbm: float = -57.0,
) -> bool:
    """校验接收机杂散发射 (8.3.3)。

    标准要求接收机杂散发射应符合国家频谱法规。
    默认限值 -57 dBm (30 MHz ~ 1 GHz 频段, GB/T 9254.1 限值)。
    """
    return spurious_dbm <= limit_dbm


# ---------------------------------------------------------------------------
# 综合校验报告
# ---------------------------------------------------------------------------
@dataclass
class RFComplianceReport:
    """射频合规综合报告。"""
    power_class: PowerClass | None = None
    power_step_ok: bool | None = None
    gfsk_dev: GFSKFreqDevResult | None = None
    evm_ok: bool | None = None
    freq_tolerance_ok: bool | None = None
    clock_ok: bool | None = None
    spurious_ok: bool | None = None
    sensitivity_dbm: float | None = None
    selectivity_ok: bool | None = None
    rssi_ok: bool | None = None
    uwb_spectrum_ok: bool | None = None
    uwb_nrmse_ok: bool | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """全部已检项均通过时返回 True。"""
        checks = [
            self.power_step_ok,
            self.gfsk_dev.passed if self.gfsk_dev else None,
            self.evm_ok,
            self.freq_tolerance_ok,
            self.clock_ok,
            self.spurious_ok,
            self.sensitivity_dbm is not None,
            self.selectivity_ok,
            self.rssi_ok,
            self.uwb_spectrum_ok,
            self.uwb_nrmse_ok,
        ]
        evaluated = [c for c in checks if c is not None]
        return all(evaluated) and len(evaluated) > 0
