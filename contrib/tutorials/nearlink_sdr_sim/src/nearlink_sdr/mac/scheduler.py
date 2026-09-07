"""时序调度器 -- TXS-10002-2025 标准 6.3/6.6/7.2。

管理 SLE 系统的基础时隙、调度时隙、超帧结构和事件组调度:

- 系统基础时隙 (Tsys = 125 μs, 30 bit 计数器)
- 系统调度时隙 (Tschedule = k × 25 μs, k ∈ {1,2,3,4,5})
- 超帧: 两个相邻 SMF 之间的时间资源
- 事件组: 多个事件的容器, 以事件组周期重复
- 时间片: 超帧内的调度时间块 (偏移量 + 持续长度 + 周期 + 重复次数)
"""

from __future__ import annotations

__all__ = [
    "EventGroupScheduler",
    "EventTimingParams",
    "LinkScheduleEntry",
    "MultiLevelInterval",
    "ScheduleManager",
    "ScheduleSlotType",
    "SleepClockAccuracy",
    "SlotCounter",
    "SmfScheduleConfig",
    "Superframe",
    "TimeSlice",
    "TxRxIntervalType",
    "schedule_slot_us",
    "tx_rx_interval_us",
]


import logging
from dataclasses import dataclass, field
from enum import IntEnum

log = logging.getLogger(__name__)

# ── 时间常量 ──

TSYS_US: int = 125
"""系统基础时隙固定值 (μs)。"""

SLOT_COUNTER_BITS: int = 30
"""时隙计数器位宽。"""

SLOT_COUNTER_MAX: int = (1 << SLOT_COUNTER_BITS) - 1
"""时隙计数器最大值 (30 bit)。"""


class ScheduleSlotType(IntEnum):
    """系统调度时隙类型枚举 (3 bit, 标准 6.3.1)。"""
    T_25US = 0    # 25 μs
    T_50US = 1    # 50 μs
    T_75US = 2    # 75 μs
    T_100US = 3   # 100 μs
    T_125US = 4   # 125 μs (默认)


# 调度时隙类型 → 实际微秒值
_SCHEDULE_SLOT_US: dict[int, int] = {
    ScheduleSlotType.T_25US: 25,
    ScheduleSlotType.T_50US: 50,
    ScheduleSlotType.T_75US: 75,
    ScheduleSlotType.T_100US: 100,
    ScheduleSlotType.T_125US: 125,
}


def schedule_slot_us(slot_type: int) -> int:
    """将调度时隙类型枚举转为微秒值。"""
    return _SCHEDULE_SLOT_US.get(slot_type, 125)


class SleepClockAccuracy(IntEnum):
    """睡眠时钟精度枚举 (3 bit)。"""
    PPM_251_500 = 0
    PPM_151_250 = 1
    PPM_101_150 = 2
    PPM_76_100 = 3
    PPM_51_75 = 4
    PPM_31_50 = 5
    PPM_21_30 = 6
    PPM_0_20 = 7


# ── 时隙计数器 ──


@dataclass
class SlotCounter:
    """30 bit 系统时隙计数器。

    以 Tsys = 125 μs 为最小递增单位, 30 bit 可表示约 37.3 小时。

    :ivar value: 当前计数值 (0 ~ 2^30 - 1)。
    """
    value: int = 0

    def advance(self, slots: int = 1) -> int:
        """前进指定时隙数, 自动回绕。"""
        self.value = (self.value + slots) & SLOT_COUNTER_MAX
        return self.value

    def time_us(self) -> int:
        """当前计数值对应的绝对时间 (μs)。"""
        return self.value * TSYS_US

    def set_from_us(self, us: int) -> None:
        """从微秒值设置计数器 (向下取整到时隙边界)。"""
        self.value = (us // TSYS_US) & SLOT_COUNTER_MAX

    def distance_to(self, target: int) -> int:
        """计算到目标时隙的距离 (考虑回绕)。"""
        return (target - self.value) & SLOT_COUNTER_MAX

    def __int__(self) -> int:
        return self.value


# ── 事件计时参数 ──


@dataclass
class EventTimingParams:
    """事件组计时参数 (标准 7.1.4.3 表29/30)。

    所有周期以调度时隙为单位, 间隔以 μs 为单位。
    """
    event_group_period: int = 0      # 事件组周期 (16 bit, 调度时隙)
    event_period: int = 0            # 事件周期 (16 bit, 调度时隙, 0=由间隔决定)
    intra_event_interval: int = 0    # 事件内间隔 (16 bit, μs)
    inter_event_interval: int = 0    # 事件间间隔/事件组间间隔 (16 bit, μs)
    event_count: int = 0             # 事件总数 (8 bit, 0=动态)
    schedule_slot_type: int = ScheduleSlotType.T_125US  # 调度时隙类型 (3 bit)
    first_tx: bool = True            # 先发指示 (True=先发, False=后发)
    tx_max_offset: int = 0           # 先发链路最大时间偏移量 (9 bit, 基础时隙)
    rx_max_offset: int = 0           # 后发链路最大时间偏移量 (9 bit, 基础时隙)
    tx_max_pdu: int = 251            # 先发链路 PDU 最大值 (11 bit, 字节)
    rx_max_pdu: int = 251            # 后发链路 PDU 最大值 (11 bit, 字节)
    delay_period: int = 0            # 延迟周期 (16 bit, 事件组周期)
    supervision_timeout: int = 100   # 超时时间 (16 bit, 10 ms)

    @property
    def schedule_slot_us(self) -> int:
        """当前调度时隙的微秒值。"""
        return schedule_slot_us(self.schedule_slot_type)

    @property
    def event_group_period_us(self) -> int:
        """事件组周期 (μs)。"""
        return self.event_group_period * self.schedule_slot_us

    @property
    def event_period_us(self) -> int:
        """事件周期 (μs), 为 0 时由事件间间隔决定。"""
        return self.event_period * self.schedule_slot_us

    @property
    def supervision_timeout_us(self) -> int:
        """超时时间 (μs)。"""
        return self.supervision_timeout * 10_000

    @property
    def max_event_duration_us(self) -> int:
        """单个事件的最大持续时间 (μs)。"""
        return (self.tx_max_offset + self.rx_max_offset + 1) * TSYS_US


# ── 时间片 ──


@dataclass
class TimeSlice:
    """超帧内时间片配置 (标准 6.6.2.2)。

    描述一个链路在超帧内的时间资源分配。

    :ivar offset: 相对 SMF 起始点的偏移 (调度时隙)。
    :ivar duration: 时间片持续长度 (调度时隙)。
    :ivar period: 时间片重复周期 (调度时隙)。
    :ivar repeat_count: 时间片重复次数。
    """
    offset: int = 0
    duration: int = 0
    period: int = 0
    repeat_count: int = 1

    def absolute_intervals(
        self, slot_us: int
    ) -> list[tuple[int, int]]:
        """计算此时间片展开后的所有绝对时间区间。

        :param slot_us: 调度时隙的微秒值。

        :returns: [(start_us, end_us), ...] 时间区间列表。
        """
        intervals: list[tuple[int, int]] = []
        for i in range(self.repeat_count):
            start = (self.offset + i * self.period) * slot_us
            end = start + self.duration * slot_us
            intervals.append((start, end))
        return intervals


# ── SMF 调度信令 ──


@dataclass
class SmfScheduleConfig:
    """SMF 调度信令参数 (标准 6.6.2.1)。

    :ivar effective_slot: 信令生效时隙 (32 bit, 基础时隙)。
    :ivar smf_interval: 两个 SMF 之间的间隔 (16 bit, 基础时隙)。
    :ivar frame_type: 无线帧类型指示 (4 bit)。
    :ivar bandwidth: 带宽指示 (2 bit, 0=1M/1=2M/2=4M)。
    :ivar pilot_density: 导频密度指示 (2 bit)。
    :ivar channel_count: 可用频点个数 (8 bit)。
    :ivar channel_table: 频点表。
    """
    effective_slot: int = 0
    smf_interval: int = 800          # 800 × 125 μs = 100 ms
    frame_type: int = 2              # FT2
    bandwidth: int = 0               # 1 MHz
    pilot_density: int = 0           # 4:1
    channel_count: int = 0
    channel_table: bytes = b""

    @property
    def smf_interval_us(self) -> int:
        """SMF 间隔 (μs)。"""
        return self.smf_interval * TSYS_US


# ── 链路调度条目 ──


@dataclass
class LinkScheduleEntry:
    """链路信令: 链路在超帧内的时间资源配置 (标准 6.6.2.2)。

    :ivar link_id: 逻辑链路标识 (24 bit)。
    :ivar effective_slot: 信令生效时隙 (32 bit, 基础时隙)。
    :ivar period_factor: 链路周期因子 (8 bit)。
    :ivar schedule_slot_type: 调度时隙长度 (3 bit)。
    :ivar time_slices: 时间片配置列表。
    """
    link_id: int = 0
    effective_slot: int = 0
    period_factor: int = 1
    schedule_slot_type: int = ScheduleSlotType.T_125US
    time_slices: list[TimeSlice] = field(default_factory=list)


# ── 超帧管理 ──


@dataclass
class Superframe:
    """超帧: 两个相邻 SMF 之间的时间资源。

    管理 SMF 周期内的链路调度, 活动区间和低功耗区间的划分。

    :ivar smf_config: SMF 调度参数。
    :ivar link_entries: 已注册的链路调度条目。
    """
    smf_config: SmfScheduleConfig = field(default_factory=SmfScheduleConfig)
    link_entries: list[LinkScheduleEntry] = field(default_factory=list)

    @property
    def duration_us(self) -> int:
        """超帧持续时间 (μs)。"""
        return self.smf_config.smf_interval_us

    @property
    def duration_slots(self) -> int:
        """超帧持续时间 (基础时隙数)。"""
        return self.smf_config.smf_interval

    def add_link(self, entry: LinkScheduleEntry) -> None:
        """注册一个链路的时间资源配置。"""
        # 移除同 link_id 的旧条目
        self.link_entries = [
            e for e in self.link_entries if e.link_id != entry.link_id
        ]
        self.link_entries.append(entry)
        log.debug(
            "链路 0x%06X 已注册, %d 个时间片",
            entry.link_id, len(entry.time_slices),
        )

    def remove_link(self, link_id: int) -> bool:
        """移除一个链路的调度配置。"""
        before = len(self.link_entries)
        self.link_entries = [
            e for e in self.link_entries if e.link_id != link_id
        ]
        return len(self.link_entries) < before

    def get_link(self, link_id: int) -> LinkScheduleEntry | None:
        """按链路标识查找调度条目。"""
        for entry in self.link_entries:
            if entry.link_id == link_id:
                return entry
        return None

    def active_region_us(self) -> tuple[int, int]:
        """计算活动区间 (从 SMF 到最后一个时间片结束)。

        :returns: (start_us, end_us) 活动区间。start_us 始终为 0 (SMF 起始)。
        """
        if not self.link_entries:
            return (0, 0)
        max_end = 0
        for entry in self.link_entries:
            slot_us = schedule_slot_us(entry.schedule_slot_type)
            for ts in entry.time_slices:
                for _start, end in ts.absolute_intervals(slot_us):
                    max_end = max(max_end, end)
        return (0, max_end)

    def check_conflicts(self) -> list[tuple[int, int, int, int]]:
        """检测时间片冲突。

        :returns: 冲突列表 [(link_id_a, link_id_b, overlap_start, overlap_end), ...]。
        """
        # 展开所有时间片
        all_intervals: list[tuple[int, int, int]] = []  # (start, end, link_id)
        for entry in self.link_entries:
            slot_us = schedule_slot_us(entry.schedule_slot_type)
            for ts in entry.time_slices:
                for start, end in ts.absolute_intervals(slot_us):
                    all_intervals.append((start, end, entry.link_id))

        all_intervals.sort()
        conflicts: list[tuple[int, int, int, int]] = []
        for i in range(len(all_intervals)):
            for j in range(i + 1, len(all_intervals)):
                s_a, e_a, lid_a = all_intervals[i]
                s_b, e_b, lid_b = all_intervals[j]
                if s_b >= e_a:
                    break  # 后续更不可能重叠
                # s_b < e_a, 有重叠
                overlap_start = max(s_a, s_b)
                overlap_end = min(e_a, e_b)
                conflicts.append((lid_a, lid_b, overlap_start, overlap_end))
        return conflicts


# ── 事件组调度器 ──


@dataclass
class EventGroupScheduler:
    """事件组调度器。

    管理一个链路上事件组的时间计算, 根据事件组参数
    生成事件的调度时间表。

    :ivar timing: 事件计时参数。
    :ivar anchor_slot: 锚点时隙 (基础时隙顺序号)。
    :ivar anchor_offset_us: 事件组起始偏移 (μs)。
    """
    timing: EventTimingParams = field(default_factory=EventTimingParams)
    anchor_slot: int = 0
    anchor_offset_us: int = 0

    def anchor_time_us(self) -> int:
        """锚点绝对时间 (μs)。"""
        return self.anchor_slot * TSYS_US + self.anchor_offset_us

    def event_start_times(
        self, group_index: int = 0
    ) -> list[int]:
        """计算指定事件组内每个事件的起始时间。

        :param group_index: 第几个事件组 (从 0 开始)。

        :returns: 事件起始时间列表 (μs, 绝对时间)。
        """
        t = self.timing
        slot_us = t.schedule_slot_us
        group_start = (
            self.anchor_time_us()
            + group_index * t.event_group_period * slot_us
        )

        count = t.event_count if t.event_count > 0 else 1
        times: list[int] = []
        for i in range(count):
            if t.event_period > 0:
                event_offset = i * t.event_period * slot_us
            else:
                event_offset = i * t.inter_event_interval
            times.append(group_start + event_offset)
        return times

    def tx_window(
        self, event_start_us: int
    ) -> tuple[int, int]:
        """计算一个事件中先发窗口的时间区间。

        :param event_start_us: 事件起始时间 (μs)。

        :returns: (start_us, end_us) 先发窗口区间。
        """
        t = self.timing
        tx_dur = (t.tx_max_offset + 1) * TSYS_US
        return (event_start_us, event_start_us + tx_dur)

    def rx_window(
        self, event_start_us: int
    ) -> tuple[int, int]:
        """计算一个事件中后发窗口的时间区间。

        :param event_start_us: 事件起始时间 (μs)。

        :returns: (start_us, end_us) 后发窗口区间。
        """
        t = self.timing
        tx_dur = (t.tx_max_offset + 1) * TSYS_US
        rx_start = event_start_us + tx_dur + t.intra_event_interval
        rx_dur = (t.rx_max_offset + 1) * TSYS_US
        return (rx_start, rx_start + rx_dur)

    def event_schedule(
        self, group_index: int = 0
    ) -> list[dict[str, int | tuple[int, int]]]:
        """生成完整的事件调度表。

        返回每个事件的起始时间、先发窗口和后发窗口。

        :param group_index: 事件组索引。

        :returns: [{"event_start": int, "tx_window": (s, e), "rx_window": (s, e)}, ...]
        """
        return [
            {
                "event_start": start,
                "tx_window": self.tx_window(start),
                "rx_window": self.rx_window(start),
            }
            for start in self.event_start_times(group_index)
        ]

    def next_event_group_start(
        self, current_us: int
    ) -> int:
        """计算下一个事件组的起始时间。

        :param current_us: 当前时间 (μs)。

        :returns: 下一个事件组起始时间 (μs)。
        """
        anchor = self.anchor_time_us()
        period = self.timing.event_group_period_us
        if period <= 0:
            return anchor

        if current_us <= anchor:
            return anchor

        elapsed = current_us - anchor
        groups_passed = elapsed // period
        next_start = anchor + (groups_passed + 1) * period
        return next_start


# ── 收发间隔管理 ──


class TxRxIntervalType(IntEnum):
    """收发间隔类型枚举 (4 bit, 标准 7.2.1)。"""
    INTERVAL_125US = 0x0
    INTERVAL_100US = 0x1
    INTERVAL_75US = 0x2
    INTERVAL_50US = 0x3
    INTERVAL_25US = 0x4


_TX_RX_INTERVAL_US: dict[int, int] = {
    TxRxIntervalType.INTERVAL_125US: 125,
    TxRxIntervalType.INTERVAL_100US: 100,
    TxRxIntervalType.INTERVAL_75US: 75,
    TxRxIntervalType.INTERVAL_50US: 50,
    TxRxIntervalType.INTERVAL_25US: 25,
}


def tx_rx_interval_us(interval_type: int) -> int:
    """收发间隔类型转微秒值。"""
    return _TX_RX_INTERVAL_US.get(interval_type, 125)


@dataclass
class MultiLevelInterval:
    """31 级多级收发间隔 (标准 7.2.2)。

    Polar 编码分段 [00001]~[11111] 各对应一个 8 bit 间隔值 (μs)。

    :ivar levels: 31 个间隔值 (μs), 索引 0-30 对应级别 1-31。
    """
    levels: list[int] = field(default_factory=lambda: [125] * 31)

    def get(self, level: int) -> int:
        """获取指定级别的间隔 (μs)。level 从 1 开始。"""
        if 1 <= level <= 31:
            return self.levels[level - 1]
        return 125

    def set(self, level: int, value_us: int) -> None:
        """设置指定级别的间隔。"""
        if 1 <= level <= 31:
            self.levels[level - 1] = value_us & 0xFF

    def pack(self) -> bytes:
        """序列化为 31 字节。"""
        return bytes(v & 0xFF for v in self.levels)

    @classmethod
    def unpack(cls, data: bytes) -> MultiLevelInterval:
        """从 31 字节反序列化。"""
        if len(data) < 31:
            raise ValueError("数据不足: 需要 31 字节")
        return cls(levels=list(data[:31]))


# ── 综合调度管理器 ──


@dataclass
class ScheduleManager:
    """综合调度管理器。

    整合超帧管理、事件组调度和收发间隔控制,
    提供链路时间资源的统一管理接口。

    :ivar superframe: 超帧结构。
    :ivar event_schedulers: 按 link_id 索引的事件组调度器。
    :ivar slot_counter: 全局时隙计数器。
    :ivar tx_rx_interval: 当前收发间隔类型。
    :ivar multi_interval: 多级收发间隔配置。
    """
    superframe: Superframe = field(default_factory=Superframe)
    event_schedulers: dict[int, EventGroupScheduler] = field(
        default_factory=dict
    )
    slot_counter: SlotCounter = field(default_factory=SlotCounter)
    tx_rx_interval: int = TxRxIntervalType.INTERVAL_125US
    multi_interval: MultiLevelInterval = field(
        default_factory=MultiLevelInterval
    )

    def configure_smf(self, config: SmfScheduleConfig) -> None:
        """配置 SMF 调度参数。"""
        self.superframe.smf_config = config
        log.info(
            "SMF 配置: 间隔=%d 时隙 (%d μs), 帧类型=%d",
            config.smf_interval, config.smf_interval_us, config.frame_type,
        )

    def register_link(
        self,
        link_id: int,
        timing: EventTimingParams,
        time_slices: list[TimeSlice] | None = None,
        anchor_slot: int = 0,
        anchor_offset_us: int = 0,
    ) -> None:
        """注册链路并配置事件组调度。

        :param link_id: 逻辑链路标识。
        :param timing: 事件计时参数。
        :param time_slices: 超帧内时间片列表 (可选)。
        :param anchor_slot: 事件组锚点时隙。
        :param anchor_offset_us: 事件组起始偏移 (μs)。
        """
        # 创建事件组调度器
        self.event_schedulers[link_id] = EventGroupScheduler(
            timing=timing,
            anchor_slot=anchor_slot,
            anchor_offset_us=anchor_offset_us,
        )

        # 注册超帧时间片
        if time_slices:
            entry = LinkScheduleEntry(
                link_id=link_id,
                effective_slot=self.slot_counter.value,
                schedule_slot_type=timing.schedule_slot_type,
                time_slices=time_slices,
            )
            self.superframe.add_link(entry)

    def unregister_link(self, link_id: int) -> None:
        """注销链路调度。"""
        self.event_schedulers.pop(link_id, None)
        self.superframe.remove_link(link_id)

    def get_event_schedule(
        self, link_id: int, group_index: int = 0
    ) -> list[dict[str, int | tuple[int, int]]] | None:
        """获取指定链路的事件调度表。"""
        sched = self.event_schedulers.get(link_id)
        if sched is None:
            return None
        return sched.event_schedule(group_index)

    def next_smf_slot(self) -> int:
        """计算下一个 SMF 时隙号。"""
        interval = self.superframe.smf_config.smf_interval
        if interval <= 0:
            return self.slot_counter.value
        current = self.slot_counter.value
        effective = self.superframe.smf_config.effective_slot
        if current <= effective:
            return effective
        elapsed = current - effective
        periods = elapsed // interval
        return (effective + (periods + 1) * interval) & SLOT_COUNTER_MAX

    def advance_time(self, slots: int) -> int:
        """推进全局时钟。"""
        return self.slot_counter.advance(slots)

    def update_tx_rx_interval(self, interval_type: int) -> int:
        """更新收发间隔类型。

        :returns: 更新后的间隔值 (μs)。
        """
        self.tx_rx_interval = interval_type
        return tx_rx_interval_us(interval_type)

    def check_supervision_timeout(self, link_id: int) -> bool:
        """检查链路是否超时。

        基于事件组调度器的超时参数判断当前时间
        是否超过了锚点 + 超时时限。

        :returns: True 表示已超时。
        """
        sched = self.event_schedulers.get(link_id)
        if sched is None:
            return False
        timeout_us = sched.timing.supervision_timeout_us
        anchor_us = sched.anchor_time_us()
        current_us = self.slot_counter.time_us()
        return (current_us - anchor_us) > timeout_us
