"""测量帧结构组装 -- TXS-10002-2025 标准 6.3.6-6.3.11.

包含半可靠组播反馈、测量帧类型 1-4 以及超宽带脉冲测量帧的结构组装功能。
"""

from __future__ import annotations

__all__ = [
    "MeasFrameConfig",
    "RadioFrameType",
    "UWBPulseConfig",
    "build_measurement_frame_1",
    "build_measurement_frame_2",
    "build_measurement_frame_3",
    "build_measurement_frame_4",
    "build_nack_feedback",
    "build_uwb_measurement_field",
    "build_uwb_pulse_measurement_frame",
    "build_uwb_sync_field",
    "equalization_guard",
]


from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from nearlink_sdr.common.m_sequence import m31_sequence, m63_sequence
from nearlink_sdr.phy.preamble import generate_preamble


class RadioFrameType(IntEnum):
    """无线帧类型, 用于确定半可靠组播反馈的 m 序列长度。"""
    FT1 = 1
    FT2 = 2
    FT3 = 3
    FT4 = 4


# ── 6.3.6 半可靠组播反馈 ──


def _nack_m_sequence(radio_ft: int, m_index: int, cyclic_shift: int) -> np.ndarray:
    """为指定无线帧类型生成 NACK 反馈 m 序列。

    :param radio_ft: 无线帧类型 (1-4)。
    :param m_index: m 序列索引。
    :param cyclic_shift: 循环移位量。

    :returns: NACK 反馈序列比特数组。
    """
    if radio_ft == 1:
        seq = m31_sequence(m_index, 31)
    elif radio_ft in (2, 3):
        half = m31_sequence(m_index, 31)
        seq = np.concatenate([half, half])
    elif radio_ft == 4:
        half = m63_sequence(m_index, 63)
        seq = np.concatenate([half, half])
    else:
        raise ValueError(f"radio_ft 必须为 1-4, 实际为 {radio_ft}")

    if cyclic_shift:
        seq = np.roll(seq, cyclic_shift)
    return seq


def build_nack_feedback(
    radio_ft: int,
    m_index: int,
    cyclic_shift: int = 0,
    symbol_rate_mhz: float = 1.0,
) -> np.ndarray:
    """构建半可靠组播 NACK 反馈帧 (标准 6.3.6)。

    帧结构: 前导信号 | 同步信号 | NACK 反馈序列

    :param radio_ft: 无线帧类型 (1-4), 决定 m 序列长度。
    :param m_index: m 序列索引, 由高层信令配置。
    :param cyclic_shift: m 序列循环移位量, 由高层信令配置。
    :param symbol_rate_mhz: 符号速率, 用于前导码生成。

    :returns: NACK 反馈序列比特数组 (不含前导和同步, 由调用者拼接)。
    """
    return _nack_m_sequence(radio_ft, m_index, cyclic_shift)


# ── 均衡保护 ──


def equalization_guard(sync_last_bit: int) -> np.ndarray:
    """生成均衡保护序列 (标准 6.3.7)。

    同步信号最后 1 比特为 1 时: 0101 (MSB 优先)
    同步信号最后 1 比特为 0 时: 1010 (MSB 优先)

    :param sync_last_bit: 同步信号的最后一个比特值 (0 或 1)。

    :returns: 4 比特均衡保护序列。
    """
    if sync_last_bit:
        return np.array([0, 1, 0, 1], dtype=np.int8)
    return np.array([1, 0, 1, 0], dtype=np.int8)


# ── 测量帧配置 ──


@dataclass
class MeasFrameConfig:
    """测量帧配置参数。"""
    sync_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    measurement_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    switch_interval_samples: int = 0
    symbol_rate_mhz: float = 1.0
    radio_frame_type: int = 2
    is_first_sender: bool = True


# ── 6.3.7 测量帧类型 1 ──


def build_measurement_frame_1(config: MeasFrameConfig) -> np.ndarray:
    """组装测量帧类型 1 (标准 6.3.7)。

    先发节点: 前导信号 | 同步信号 | 均衡保护 | 切换间隔 | 测量信号
    后发节点: 测量信号 | 切换间隔 | 前导信号 | 同步信号 | 均衡保护

    :param config: 测量帧配置。

    :returns: 组装后的帧比特/符号序列。
    """
    preamble = generate_preamble(config.radio_frame_type, config.symbol_rate_mhz)
    sync = config.sync_signal
    guard = equalization_guard(int(sync[-1]) if len(sync) > 0 else 0)
    switch = np.zeros(config.switch_interval_samples, dtype=np.int8)
    meas = config.measurement_signal

    if config.is_first_sender:
        return np.concatenate([preamble, sync, guard, switch, meas])
    return np.concatenate([meas, switch, preamble, sync, guard])


# ── 6.3.8 测量帧类型 2 ──


def build_measurement_frame_2(config: MeasFrameConfig) -> np.ndarray:
    """组装测量帧类型 2 (标准 6.3.8)。

    仅包含测量信号。

    :param config: 测量帧配置。

    :returns: 测量信号序列。
    """
    return config.measurement_signal.copy()


# ── 6.3.9 测量帧类型 3 ──


def build_measurement_frame_3(config: MeasFrameConfig) -> np.ndarray:
    """组装测量帧类型 3 (标准 6.3.9)。

    先发节点: 前导信号 | 同步信号 | 均衡保护
    后发节点: 前导信号 | 同步信号 | 均衡保护 | 切换间隔 | 测量信号

    用于位置测量事件组的初始化阶段。

    :param config: 测量帧配置。

    :returns: 组装后的帧比特/符号序列。
    """
    preamble = generate_preamble(config.radio_frame_type, config.symbol_rate_mhz)
    sync = config.sync_signal
    guard = equalization_guard(int(sync[-1]) if len(sync) > 0 else 0)

    if config.is_first_sender:
        return np.concatenate([preamble, sync, guard])

    switch = np.zeros(config.switch_interval_samples, dtype=np.int8)
    meas = config.measurement_signal
    return np.concatenate([preamble, sync, guard, switch, meas])


# ── 6.3.10 测量帧类型 4 ──


def build_measurement_frame_4(config: MeasFrameConfig) -> np.ndarray:
    """组装测量帧类型 4 (标准 6.3.10)。

    先发/后发节点结构相同: 前导信号 | 同步信号 | 均衡保护 | 切换间隔 | 测量信号

    用于超宽带脉冲测量的初始化同步阶段,
    测量信号应为窄带波形测量信号 1 或 N=1 的窄带波形测量信号 2。

    :param config: 测量帧配置。

    :returns: 组装后的帧比特/符号序列。
    """
    preamble = generate_preamble(config.radio_frame_type, config.symbol_rate_mhz)
    sync = config.sync_signal
    guard = equalization_guard(int(sync[-1]) if len(sync) > 0 else 0)
    switch = np.zeros(config.switch_interval_samples, dtype=np.int8)
    meas = config.measurement_signal
    return np.concatenate([preamble, sync, guard, switch, meas])


# ── 6.3.11 超宽带脉冲测量帧 ──


@dataclass
class UWBPulseConfig:
    """超宽带脉冲测量帧配置。

    :ivar K: 码字长度。
    :ivar L: 占空因子。
    :ivar Tc: 码片时长 (秒)。
    :ivar N_sync: 同步字段符号个数, 0 表示不发送同步字段。
    :ivar symbol_seq: 同步符号序列 (+1/-1), 长度为 K。
    :ivar M_seg: 测量子片段数。
    :ivar N_seg: 每个测量子片段的 CTS 符号数。
    :ivar N_gap: 测量子片段间隔的符号数。
    :ivar scramble: 加扰 SC 序列 (+1/-1), 长度为 N_seg * M_seg。
        普通模式下全 1, 安全模式下由安全算法确定。
    :ivar L_cp: 循环前缀码片数 (安全模式)。
    :ivar L_zero: 补零后缀码片数 (安全模式)。
    :ivar N_offset: 安全模式偏移码片数, 默认 0 (普通模式)。
    """
    K: int = 31
    L: int = 4
    Tc: float = 1e-9
    N_sync: int = 8
    symbol_seq: np.ndarray = field(default_factory=lambda: np.ones(31, dtype=np.float64))
    M_seg: int = 1
    N_seg: int = 1
    N_gap: int = 4
    scramble: np.ndarray | None = None
    L_cp: int = 0
    L_zero: int = 0
    N_offset: int = 0


def _apply_duty_cycle(symbol: np.ndarray, L: int) -> np.ndarray:
    """对码字序列进行占空比为 1/L 的插零时域扩展。"""
    expanded = np.zeros(len(symbol) * L, dtype=symbol.dtype)
    expanded[::L] = symbol
    return expanded


def build_uwb_sync_field(config: UWBPulseConfig) -> np.ndarray:
    """构建超宽带脉冲测量帧同步字段 (标准 6.3.11.1)。

    同步字段由 N_sync 个相同符号重复组成, 每个符号经过插零时域扩展。
    总长度: N_sync * K * L 个码片。

    :param config: UWB 脉冲配置。

    :returns: 同步字段码片序列。
    """
    if config.N_sync == 0:
        return np.array([], dtype=np.float64)

    one_symbol = _apply_duty_cycle(config.symbol_seq[:config.K], config.L)
    return np.tile(one_symbol, config.N_sync)


def _build_cts_symbol(
    symbol: np.ndarray,
    L: int,
    scramble_val: float,
    L_cp: int = 0,
    L_zero: int = 0,
) -> np.ndarray:
    """构建单个 CTS 测量符号。

    普通模式: 直接插零扩展。
    安全模式 (L_cp>0 或 L_zero>0): 插零扩展后乘以加扰,
    增加循环前缀和补零后缀。
    """
    base = _apply_duty_cycle(symbol, L) * scramble_val

    if L_cp == 0 and L_zero == 0:
        return base

    cp = base[-L_cp:] if L_cp > 0 else np.array([], dtype=base.dtype)
    suffix = np.zeros(L_zero, dtype=base.dtype)
    return np.concatenate([cp, base, suffix])


def build_uwb_measurement_field(
    config: UWBPulseConfig,
    cts_symbol_seq: np.ndarray | None = None,
) -> np.ndarray:
    """构建超宽带脉冲测量帧测量字段 (标准 6.3.11.2)。

    测量字段由 M_seg 个测量子片段组成, 每个子片段包含 N_seg 个 CTS 符号,
    相邻子片段之间有时间间隔。

    :param config: UWB 脉冲配置。
    :param cts_symbol_seq: CTS 测量符号序列, 默认使用 config.symbol_seq。

    :returns: 测量字段码片序列。
    """
    N_cts = config.N_seg * config.M_seg
    if N_cts == 0:
        return np.array([], dtype=np.float64)

    sym = cts_symbol_seq if cts_symbol_seq is not None else config.symbol_seq[:config.K]
    scramble = config.scramble
    if scramble is None:
        scramble = np.ones(N_cts, dtype=np.float64)

    # 符号间隔 (码片数)
    T_bsym = config.K * config.L
    gap_chips = config.N_gap * T_bsym - config.N_offset

    segments: list[np.ndarray] = []
    cts_idx = 0
    for seg_i in range(config.M_seg):
        if seg_i > 0 and gap_chips > 0:
            segments.append(np.zeros(gap_chips, dtype=np.float64))
        for _ in range(config.N_seg):
            cts = _build_cts_symbol(
                sym, config.L, float(scramble[cts_idx]),
                config.L_cp, config.L_zero,
            )
            segments.append(cts)
            cts_idx += 1

    return np.concatenate(segments)


def build_uwb_pulse_measurement_frame(
    config: UWBPulseConfig,
    cts_symbol_seq: np.ndarray | None = None,
) -> np.ndarray:
    """构建完整的超宽带脉冲测量帧 (标准 6.3.11)。

    帧结构: 同步字段 | 测量字段

    :param config: UWB 脉冲配置。
    :param cts_symbol_seq: CTS 测量符号序列, 默认使用同步符号序列。

    :returns: 完整帧码片序列。
    """
    sync = build_uwb_sync_field(config)
    meas = build_uwb_measurement_field(config, cts_symbol_seq)
    return np.concatenate([sync, meas])
