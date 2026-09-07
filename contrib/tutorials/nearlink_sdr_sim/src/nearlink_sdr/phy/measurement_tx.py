"""测量链路传输与测量量计算 -- TXS-10002-2025 标准 6.7/6.8。

实现窄带跳频和超宽带脉冲测量链路的参数配置、时间资源调度和测量量计算:
  - 6.7.2/6.7.3: 测量/感知链路参数模型
  - 6.7.4: 时间资源配置 (事件组调度)
  - 6.7.5/6.8.5: 测量量 (飞行时间/角度/CIR/距离多普勒)
"""

from __future__ import annotations

__all__ = [
    "UWB_CONFIG_TIME_GRANULARITY_TC",
    "AntennaOrderType",
    "CIRConfig",
    "CSIFeedback",
    "ChannelSpliceMode",
    "EventFrameType",
    "HoppingOrder",
    "MeasBandwidth",
    "MeasDirection",
    "MeasLinkParams",
    "NodeMeasConfig",
    "SecurityType",
    "TimeRefType",
    "UWBMeasLinkParams",
    "UWBMeasMode",
    "angle_estimate",
    "compute_csi_feedback",
    "csi_to_rx_power",
    "ds_twr_2msg",
    "ds_twr_3msg",
    "event_schedule",
    "event_start_times",
    "extract_cir",
    "range_doppler",
    "uwb_event_count_per_mode",
    "uwb_event_sender",
]


import cmath
import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------


class MeasDirection(IntEnum):
    """测量方向。"""
    UNIDIRECTIONAL = 0    # 单向测量
    BIDIRECTIONAL = 1     # 双向测量


class HoppingOrder(IntEnum):
    """跳频方式 (6.7.2)。"""
    LOW_TO_HIGH = 0       # 信道号从低到高
    HIGH_TO_LOW = 1       # 信道号从高到低
    ALGORITHMIC = 2       # 按跳频算法


class MeasBandwidth(IntEnum):
    """测量信号带宽 (6.7.2)。"""
    BW_1MHZ = 0
    BW_2MHZ = 1
    BW_4MHZ = 2


class SecurityType(IntEnum):
    """安全类型。"""
    NONE = 0
    SECURE = 1


class TimeRefType(IntEnum):
    """初始时间参考类型 (6.8.5.4)。"""
    FIRST_PATH = 0        # 第一径到达时间
    STRONGEST_PATH = 1    # 最强径到达时间
    TX_TIME = 2           # 信号发送时间


class AntennaOrderType(IntEnum):
    """天线对次序类型 (6.7.2)。"""
    SEQUENTIAL = 0
    RANDOM = 1


class EventFrameType(IntEnum):
    """事件中传输的帧类型。"""
    INIT = 0              # 初始化阶段事件
    TYPE_1 = 1            # 测量帧类型 1
    TYPE_2 = 2            # 测量帧类型 2


# ---------------------------------------------------------------------------
# 6.7.2/6.7.3 测量/感知链路参数
# ---------------------------------------------------------------------------


@dataclass
class NodeMeasConfig:
    """单个节点的测量参数。"""
    sync_signal_type: int = 0       # 同步信号类型
    sync_signal_length: int = 64    # 同步信号长度
    meas_signal_length: int = 128   # 测量序列长度
    switch_interval_us: float = 0   # 同步-测量切换间隔 (us)
    ft1_duration_us: float = 200    # 测量帧类型1传输时长 (us)
    ft2_duration_us: float = 100    # 测量帧类型2传输时长 (us)
    security_type: SecurityType = SecurityType.NONE
    antenna_count: int = 1


@dataclass
class MeasLinkParams:
    """测量/感知链路参数 (6.7.2/6.7.3)。"""
    # 事件组时间参数
    event_group_start_us: float = 0       # 事件组起始时刻 (us)
    event_group_period_us: float = 10000  # 事件组周期 (us)
    ft1_inter_event_us: float = 500       # 类型1事件间间隔 (us)
    ft2_inter_event_us: float = 300       # 类型2事件间间隔 (us)
    init_inter_event_us: float = 400      # 初始化阶段事件间间隔 (us)
    intra_event_us: float = 100           # 事件内间隔 (us)
    inter_event_us: float = 200           # 事件间间隔 (us)
    inter_group_us: float = 1000          # 事件组间间隔 (us)
    event_count: int = 4                  # 事件总数/事件组
    ft1_period: int = 2                   # 类型1事件周期
    has_init_phase: bool = False          # 是否存在初始化阶段

    # 方向与跳频
    direction: MeasDirection = MeasDirection.UNIDIRECTIONAL
    first_sender: int = 0                 # 先发后发指示 (0=配置方先发)
    hopping_order: HoppingOrder = HoppingOrder.LOW_TO_HIGH
    bandwidth: MeasBandwidth = MeasBandwidth.BW_1MHZ
    init_channel: int = 0                 # 初始化频点信道号

    # 先发/后发节点配置
    first_node: NodeMeasConfig = field(default_factory=NodeMeasConfig)
    second_node: NodeMeasConfig = field(default_factory=NodeMeasConfig)

    # 多天线参数
    antenna_switch_interval_us: float = 10  # 天线切换间隔 (us)
    first_sub_signal_length: int = 0        # 第一个子测量信号长度
    antenna_order_type: AntennaOrderType = AntennaOrderType.SEQUENTIAL
    antenna_random_k: int = 0               # 天线随机数位宽

    # 安全
    security_random_seed: int = 0

    @property
    def has_multi_antenna(self) -> bool:
        return (self.first_node.antenna_count > 1
                or self.second_node.antenna_count > 1)


# ---------------------------------------------------------------------------
# 6.7.4 时间资源配置
# ---------------------------------------------------------------------------


def event_schedule(params: MeasLinkParams) -> list[tuple[int, EventFrameType]]:
    """生成一个事件组内的事件调度序列。

    :param params: 测量链路参数。

    :returns: 列表, 每个元素为 (事件索引, 帧类型)。
    """
    schedule: list[tuple[int, EventFrameType]] = []

    for idx in range(params.event_count):
        if params.has_init_phase and idx == 0:
            schedule.append((idx, EventFrameType.INIT))
            continue

        effective_idx = idx

        if params.ft1_period == 0:
            # 所有事件都是类型2
            schedule.append((idx, EventFrameType.TYPE_2))
        elif params.ft1_period == 1:
            schedule.append((idx, EventFrameType.TYPE_1))
        else:
            # 初始化占了第0个, 其余按周期分
            if params.has_init_phase:
                # 索引0已是INIT, 其余看 effective_idx 是否为 ft1_period 倍数
                if effective_idx % params.ft1_period == 0:
                    schedule.append((idx, EventFrameType.TYPE_1))
                else:
                    schedule.append((idx, EventFrameType.TYPE_2))
            else:
                if effective_idx % params.ft1_period == 0:
                    schedule.append((idx, EventFrameType.TYPE_1))
                else:
                    schedule.append((idx, EventFrameType.TYPE_2))

    return schedule


def event_start_times(
    params: MeasLinkParams,
    group_index: int = 0,
) -> list[float]:
    """计算一个事件组内各事件的起始时刻 (us)。

    :param params: 测量链路参数。
    :param group_index: 事件组序号。

    :returns: 各事件的起始时刻列表 (us)。
    """
    schedule = event_schedule(params)
    group_offset = (params.event_group_start_us
                    + group_index * params.event_group_period_us)

    times: list[float] = []
    current = group_offset

    for i, (_, ft) in enumerate(schedule):
        times.append(current)
        if i < len(schedule) - 1:
            if ft == EventFrameType.INIT:
                gap = params.init_inter_event_us
            elif ft == EventFrameType.TYPE_1:
                gap = params.ft1_inter_event_us
            else:
                gap = params.ft2_inter_event_us
            current += gap

    return times


# ---------------------------------------------------------------------------
# 6.8.5.1 双边两消息测量量 (DS-TWR 2-message)
# ---------------------------------------------------------------------------


def ds_twr_2msg(ra: float, db: float) -> float:
    """双边两消息飞行时间估计 (6.8.5.1)。

    :param ra: 先发节点测量的往返时间差 Ra = Ra2 - Ra1。
    :param db: 后发节点测量的处理时间差 Db = Db2 - Db1。

    :returns: 单向飞行时间估计 T_prop (与 ra/db 相同时间单位)。
    """
    return 0.5 * (ra - db)


# ---------------------------------------------------------------------------
# 6.8.5.2 双边三消息测量量 (DS-TWR 3-message)
# ---------------------------------------------------------------------------


def ds_twr_3msg(ra: float, rb: float, da: float, db: float) -> float:
    """双边三消息飞行时间估计 (6.8.5.2)。

    :param ra: 先发节点 Ra = Ra2 - Ra1。
    :param rb: 后发节点 Rb = Rb2 - Rb1。
    :param da: 先发节点 Da = Da2 - Da1。
    :param db: 后发节点 Db = Db2 - Db1。

    :returns: 单向飞行时间估计 T_prop。
    """
    numerator = ra * rb - da * db
    denominator = ra + rb + da + db
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# 6.8.5.3 角度测量量 (AoA/AoD)
# ---------------------------------------------------------------------------


def angle_estimate(
    delta_phi: float,
    wavelength: float,
    antenna_spacing: float,
) -> float:
    """根据相位差估计到达角/出发角 (6.8.5.3)。

    :param delta_phi: 两天线接收/发送信号的载波相位差 (rad)。
    :param wavelength: 信号波长 (m)。
    :param antenna_spacing: 天线间距 (m)。

    :returns: 角度估计值 (rad), 范围 [0, pi]。
    """
    cos_val = (delta_phi * wavelength) / (2.0 * math.pi * antenna_spacing)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.acos(cos_val)


# ---------------------------------------------------------------------------
# 6.8.5.4 信道冲击响应测量量 (CIR)
# ---------------------------------------------------------------------------


@dataclass
class CIRConfig:
    """信道冲击响应测量配置。"""
    time_ref: TimeRefType = TimeRefType.FIRST_PATH
    delay_offset_ns: float = 0.0      # 多径时延偏移量 (ns)
    delay_length_ns: float = 100.0    # 多径时延指示长度 (ns)
    sample_rate_hz: float = 499.2e6   # 时域采样频率


def extract_cir(
    rx_signal: NDArray[np.complex128],
    config: CIRConfig,
    ref_sample: int = 0,
) -> NDArray[np.complex128]:
    """从接收信号中提取信道冲击响应测量量 (6.8.5.4)。

    :param rx_signal: 接收到的时域 IQ 信号。
    :param config: CIR 测量配置。
    :param ref_sample: 参考时间点对应的采样索引。

    :returns: 截取的 CIR (I+jQ 复数数组)。
    """
    samples_per_ns = config.sample_rate_hz / 1e9
    offset_samples = int(config.delay_offset_ns * samples_per_ns)
    length_samples = max(1, int(config.delay_length_ns * samples_per_ns))

    start = ref_sample + offset_samples
    start = max(0, start)
    end = min(len(rx_signal), start + length_samples)

    return rx_signal[start:end].copy()


# ---------------------------------------------------------------------------
# 6.8.5.5 距离多普勒测量量
# ---------------------------------------------------------------------------


def range_doppler(
    cir_frames: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """距离多普勒测量量计算 (6.8.5.5)。

    对多帧 CIR 数据在慢时间维度做 FFT 获取距离-多普勒图。

    :param cir_frames: 形状 (n_frames, n_delay_bins) 的多帧 CIR 数据。

    :returns: 形状 (n_frames, n_delay_bins) 的距离-多普勒矩阵。
    """
    if cir_frames.ndim == 1:
        return np.fft.fft(cir_frames)
    # 沿帧维度 (慢时间, axis=0) 做 FFT
    return np.fft.fft(cir_frames, axis=0)


# ---------------------------------------------------------------------------
# CSI 反馈量化 (6.7.5)
# ---------------------------------------------------------------------------


@dataclass
class CSIFeedback:
    """信道状态信息反馈 (6.7.5)。

    接收信号功率(dBm) = 20*lg(abs(相对IQ值/1024)) + 参考接收功率值
    """
    ref_power_dbm: float         # 参考接收功率值 (dBm)
    relative_iq: complex         # 相对 IQ 值 (线性复数)


def compute_csi_feedback(
    rx_power_dbm: float,
    channel_iq: complex,
) -> CSIFeedback:
    """从接收功率和信道 IQ 估计值生成 CSI 反馈。

    :param rx_power_dbm: 接收信号功率 (dBm)。
    :param channel_iq: 信道估计的复数 IQ 值。

    :returns: CSIFeedback 实例。
    """
    iq_mag = abs(channel_iq)
    if iq_mag < 1e-30:
        return CSIFeedback(ref_power_dbm=rx_power_dbm, relative_iq=0j)

    # 选择 ref_power 使 relative_iq 量化精度最好
    # 标准: rx_power = 20*lg(|rel_iq|/1024) + ref_power
    # 取 ref_power = rx_power, 则 |rel_iq| = 1024
    # 保持相位: rel_iq = 1024 * exp(j*angle)
    phase = cmath.phase(channel_iq)
    relative_iq = 1024.0 * cmath.exp(1j * phase)

    return CSIFeedback(
        ref_power_dbm=rx_power_dbm,
        relative_iq=relative_iq,
    )


def csi_to_rx_power(feedback: CSIFeedback) -> float:
    """从 CSI 反馈恢复接收信号功率。

    :param feedback: CSI 反馈数据。

    :returns: 接收信号功率 (dBm)。
    """
    iq_mag = abs(feedback.relative_iq)
    if iq_mag < 1e-30:
        return feedback.ref_power_dbm - 200.0
    return 20.0 * math.log10(iq_mag / 1024.0) + feedback.ref_power_dbm


# ---------------------------------------------------------------------------
# 6.8.3/6.8.4 超宽带脉冲测量链路参数与配置
# ---------------------------------------------------------------------------


class UWBMeasMode(IntEnum):
    """超宽带脉冲测量模式 (6.8.1)。"""
    ONE_WAY = 0           # 单向
    DS_TWR_2MSG = 1       # 双边两消息
    DS_TWR_3MSG = 2       # 双边三消息


class ChannelSpliceMode(IntEnum):
    """信道拼接方式 (6.8.3)。"""
    NONE = 0              # 不拼接
    OVERLAP = 1           # 重叠方式
    CONTINUOUS = 2        # 连续方式
    NON_CONTINUOUS = 3    # 非连续方式


UWB_CONFIG_TIME_GRANULARITY_TC = 256  # 超宽带脉冲配置时间粒度 = 256*Tc


@dataclass
class UWBMeasLinkParams:
    """超宽带脉冲测量链路参数 (6.8.3/6.8.4)。"""
    mode: UWBMeasMode = UWBMeasMode.DS_TWR_2MSG
    event_group_period_us: float = 10000
    event_count: int = 4
    intra_event_us: float = 100         # 事件内间隔
    inter_event_us: float = 500         # 事件间间隔
    inter_group_us: float = 2000        # 事件组间间隔
    first_sender: int = 0               # 0=配置方先发
    alternate_sender: bool = False      # 交替先发标志

    # 信道
    channels: list[int] = field(default_factory=lambda: [0])
    splice_mode: ChannelSpliceMode = ChannelSpliceMode.NONE

    # 符号
    code_length: int = 31              # 码字长度
    cyclic_shift: int = 0              # 码字循环移位

    # 多天线
    tx_antenna_count: int = 1
    rx_antenna_count: int = 1
    antenna_switch_start_seg: int = 0  # 天线切换起始测量子片段
    sub_segment_gap_us: float = 5.0    # 测量子片段间间隔

    # 安全
    secure_mode: bool = False

    # 聚合上报
    aggregate_events: int = 1          # 聚合多少个事件后上报

    @property
    def total_antenna_pairs(self) -> int:
        return self.tx_antenna_count * self.rx_antenna_count


def uwb_event_sender(
    params: UWBMeasLinkParams,
    event_index: int,
) -> int:
    """确定指定事件中谁先发。

    :param params: UWB 测量参数。
    :param event_index: 事件索引。

    :returns: 0=配置方先发, 1=对端先发。
    """
    if not params.alternate_sender:
        return params.first_sender
    return (params.first_sender + event_index) % 2


def uwb_event_count_per_mode(mode: UWBMeasMode) -> int:
    """每个测量事件中包含的 UWB 帧数。"""
    if mode == UWBMeasMode.ONE_WAY:
        return 1
    if mode == UWBMeasMode.DS_TWR_2MSG:
        return 2
    return 3  # DS_TWR_3MSG
