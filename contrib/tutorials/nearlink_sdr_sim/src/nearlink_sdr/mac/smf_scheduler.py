"""系统管理帧发送调度 — TXS-10002-2025 标准 6.6.3。

实现系统管理帧的自身调度逻辑:
  - 6.6.3.1.1  通过接入启动调度: 从接入基本信息提取 SMF 配置参数
  - 6.6.3.1.2  通过广播启动调度: 从扩展广播帧提取 SMF 配置参数
  - SMF 周期性发送计划
  - SMF 帧构建 (含调度信令 + 链路信令 + 偏移信令)
"""

from __future__ import annotations

__all__ = [
    "SMFActivationSource",
    "SMFScheduleParams",
    "SMFTransmission",
    "SMFTransmitScheduler",
    "smf_params_from_access",
    "smf_params_from_broadcast",
]


import logging
from dataclasses import dataclass, field
from enum import IntEnum

from nearlink_sdr.mac.smf import (
    LinkSignaling,
    OffsetSignaling,
    OffsetUnit,
    ScheduleSignaling,
    SegmentIndication,
    SMFHeader,
    SystemManagementFrame,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMF 调度来源
# ---------------------------------------------------------------------------
class SMFActivationSource(IntEnum):
    """SMF 调度激活来源 (6.6.3.1)。"""
    ACCESS = 0        # 通过接入启动 (6.6.3.1.1)
    BROADCAST = 1     # 通过广播启动 (6.6.3.1.2)


# ---------------------------------------------------------------------------
# SMF 调度配置
# ---------------------------------------------------------------------------
@dataclass
class SMFScheduleParams:
    """SMF 调度全量参数, 从接入或广播过程提取。

    通过接入启动时, 这些参数来自接入请求/响应帧中的接入基本信息;
    通过广播启动时, 来自扩展广播帧中启动系统管理帧信息 (7.1.4.7)。
    """
    source: SMFActivationSource = SMFActivationSource.ACCESS
    baseline_slot: int = 0       # 系统管理帧基线时隙 / 起始时隙
    offset_us: int = 0           # 启动系统管理帧的时间偏移量 (us)
    interval: int = 800          # 系统管理帧间隔 (基础时隙单位)
    llid: int = 0x000001         # 系统管理帧逻辑链路标识
    access_addr: int = 0         # 接入地址 (广播模式特有)
    frame_type: int = 0          # 无线帧类型 (4 bits)
    bandwidth: int = 0           # 带宽指示 (0=1M, 1=2M, 2=4M)
    pilot_density: int = 0       # 导频密度 (0=4:1, 1=8:1, 2=16:1, 3=无)
    channel_count: int = 3       # 可用频点个数
    channel_table: bytes = b"\x00\x01\x02"  # 跳频表


def smf_params_from_access(
    smf_baseline_slot: int,
    smf_offset: int,
    smf_link_id: int,
    smf_period: int,
    smf_frame_type: int,
    smf_bandwidth: int,
    smf_pilot_density: int,
    smf_channel_count: int,
    smf_channel_table: bytes,
) -> SMFScheduleParams:
    """从接入基本信息中提取 SMF 调度参数 (6.6.3.1.1)。"""
    return SMFScheduleParams(
        source=SMFActivationSource.ACCESS,
        baseline_slot=smf_baseline_slot,
        offset_us=smf_offset,
        interval=smf_period,
        llid=smf_link_id,
        frame_type=smf_frame_type,
        bandwidth=smf_bandwidth,
        pilot_density=smf_pilot_density,
        channel_count=smf_channel_count,
        channel_table=smf_channel_table,
    )


def smf_params_from_broadcast(
    baseline_slot: int,
    offset: int,
    access_addr: int,
    period: int,
    frame_type: int,
    bandwidth: int,
    pilot_density: int,
    channel_count: int,
    channel_table: bytes,
) -> SMFScheduleParams:
    """从扩展广播帧的系统管理帧信息中提取 SMF 调度参数 (6.6.3.1.2)。"""
    return SMFScheduleParams(
        source=SMFActivationSource.BROADCAST,
        baseline_slot=baseline_slot,
        offset_us=offset,
        access_addr=access_addr,
        interval=period,
        frame_type=frame_type,
        bandwidth=bandwidth,
        pilot_density=pilot_density,
        channel_count=channel_count,
        channel_table=channel_table,
    )


# ---------------------------------------------------------------------------
# SMF 发送计划
# ---------------------------------------------------------------------------
@dataclass
class SMFTransmission:
    """一次 SMF 发送实例。"""
    slot: int                   # 绝对时隙号
    channel_index: int          # 频点索引 (channel_table 中的位置)
    frame: SystemManagementFrame | None = None


@dataclass
class SMFTransmitScheduler:
    """SMF 发送调度器。

    管理 SMF 的周期性发送计划:
    1. 根据参数计算各 SMF 的发送时隙
    2. 按跳频表轮换频点
    3. 构建包含调度/链路/偏移信令的 SMF 帧
    """
    params: SMFScheduleParams = field(default_factory=SMFScheduleParams)
    _next_seq: int = 0     # 信令编号计数
    _hop_index: int = 0    # 频点轮换索引

    # 已注册的链路信令 (link_id -> LinkSignaling)
    _links: dict[int, LinkSignaling] = field(default_factory=dict)
    # 已注册的偏移信令 (link_id -> OffsetSignaling)
    _offsets: dict[int, OffsetSignaling] = field(default_factory=dict)
    # 上次发送时隙
    _last_tx_slot: int = -1
    # 调度信令更新标记
    _schedule_dirty: bool = True

    def configure(self, params: SMFScheduleParams) -> None:
        """配置 SMF 参数。"""
        self.params = params
        self._schedule_dirty = True
        self._hop_index = 0
        self._next_seq = 0
        self._last_tx_slot = -1
        log.info(
            "SMF 调度配置: source=%s, baseline=%d, interval=%d, "
            "frame_type=%d, bw=%d, channels=%d",
            params.source.name, params.baseline_slot,
            params.interval, params.frame_type,
            params.bandwidth, params.channel_count,
        )

    def register_link(self, sig: LinkSignaling) -> None:
        """注册链路信令, 下次 SMF 帧将携带该信令。"""
        self._links[sig.llid] = sig
        self._schedule_dirty = True

    def unregister_link(self, llid: int) -> None:
        """移除链路信令。"""
        self._links.pop(llid, None)
        self._offsets.pop(llid, None)
        self._schedule_dirty = True

    def register_offset(self, sig: OffsetSignaling) -> None:
        """注册偏移信令。"""
        self._offsets[sig.llid] = sig
        self._schedule_dirty = True

    def next_smf_slot(self, current_slot: int) -> int:
        """计算从 current_slot 开始的下一个 SMF 发送时隙。"""
        interval = self.params.interval
        if interval <= 0:
            return current_slot
        base = self.params.baseline_slot
        if current_slot < base:
            return base
        elapsed = current_slot - base
        periods = elapsed // interval
        candidate = base + (periods + 1) * interval
        return candidate

    def smf_slot_sequence(
        self, start_slot: int, count: int,
    ) -> list[int]:
        """生成 count 个连续 SMF 发送时隙。"""
        slots: list[int] = []
        current = start_slot
        for _ in range(count):
            s = self.next_smf_slot(current)
            slots.append(s)
            current = s + 1
        return slots

    def _next_channel(self) -> int:
        """按轮换获取下一个频点索引。"""
        if self.params.channel_count <= 0:
            return 0
        idx = self._hop_index % self.params.channel_count
        self._hop_index += 1
        return idx

    def _alloc_seq(self) -> int:
        """分配信令编号 (0-63 循环)。"""
        seq = self._next_seq % 64
        self._next_seq += 1
        return seq

    def build_smf_frame(
        self,
        include_schedule: bool = True,
    ) -> SystemManagementFrame:
        """构建一个 SMF 帧。

        :param include_schedule: 是否包含调度信令 (通常首帧或参数变更时携带)。

        :returns: 包含当前所有已注册信令的 SystemManagementFrame。
        """
        frame = SystemManagementFrame(
            header=SMFHeader(
                segment_indication=SegmentIndication.COMPLETE,
                signaling_number=self._alloc_seq(),
            ),
        )

        if include_schedule or self._schedule_dirty:
            sched = ScheduleSignaling(
                effective_slot=self.params.baseline_slot,
                interval=self.params.interval,
                frame_type=self.params.frame_type,
                bandwidth=self.params.bandwidth,
                pilot_density=self.params.pilot_density,
                channel_count=self.params.channel_count,
                channel_table=self.params.channel_table,
            )
            frame.add_schedule(sched)
            self._schedule_dirty = False

        for sig in self._links.values():
            frame.add_link(sig)

        for sig in self._offsets.values():
            frame.add_offset(sig)

        return frame

    def schedule_transmission(
        self, current_slot: int,
    ) -> SMFTransmission:
        """计划下一次 SMF 发送。

        返回包含目标时隙、频点索引和帧内容的发送实例。
        """
        slot = self.next_smf_slot(current_slot)
        ch = self._next_channel()
        include_sched = (
            self._schedule_dirty
            or self._last_tx_slot < 0
        )
        frame = self.build_smf_frame(include_schedule=include_sched)
        self._last_tx_slot = slot
        return SMFTransmission(
            slot=slot,
            channel_index=ch,
            frame=frame,
        )

    def update_schedule(
        self,
        new_interval: int | None = None,
        new_frame_type: int | None = None,
        new_bandwidth: int | None = None,
        new_pilot_density: int | None = None,
        new_channel_table: bytes | None = None,
        effective_slot: int | None = None,
    ) -> ScheduleSignaling:
        """更新 SMF 调度参数, 返回新的调度信令。

        参数变更将在 effective_slot 指定的时隙生效。
        """
        if new_interval is not None:
            self.params.interval = new_interval
        if new_frame_type is not None:
            self.params.frame_type = new_frame_type
        if new_bandwidth is not None:
            self.params.bandwidth = new_bandwidth
        if new_pilot_density is not None:
            self.params.pilot_density = new_pilot_density
        if new_channel_table is not None:
            self.params.channel_table = new_channel_table
            self.params.channel_count = len(new_channel_table)
        if effective_slot is not None:
            self.params.baseline_slot = effective_slot

        self._schedule_dirty = True

        return ScheduleSignaling(
            effective_slot=self.params.baseline_slot,
            interval=self.params.interval,
            frame_type=self.params.frame_type,
            bandwidth=self.params.bandwidth,
            pilot_density=self.params.pilot_density,
            channel_count=self.params.channel_count,
            channel_table=self.params.channel_table,
        )

    def compute_offset_us(
        self,
        smf_start_slot: int,
        link_first_slot: int,
        base_slot_us: int = 125,
        unit: int = OffsetUnit.US_25,
    ) -> OffsetSignaling:
        """计算从 SMF 起始点到链路起始点的偏移量 (6.6.2.3)。

        :param smf_start_slot: SMF 起始时隙。
        :param link_first_slot: 链路第一个起始时隙。
        :param base_slot_us: 基础时隙长度 (us)。
        :param unit: 偏移量单位。

        :returns: OffsetSignaling 实例。
        """
        delta_us = (link_first_slot - smf_start_slot) * base_slot_us
        unit_us = 25 if unit == OffsetUnit.US_25 else 300
        offset_val = delta_us // unit_us
        if offset_val < 0:
            offset_val = 0
        if offset_val > 0x7FFF:
            offset_val = 0x7FFF
        return OffsetSignaling(
            llid=0,
            offset=offset_val,
            offset_unit=unit,
        )
