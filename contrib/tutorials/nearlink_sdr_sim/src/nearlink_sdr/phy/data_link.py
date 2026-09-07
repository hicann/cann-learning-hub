"""异步/同步数据链路传输规程 (标准 6.5.1 ~ 6.5.3)。

实现数据链路参数管理、流控决策、同步数据丢弃机制、
事件组集合调度, 以及同步业务适配 (周期/非周期) 的
分段重组逻辑。
"""

from __future__ import annotations

__all__ = [
    "AdaptMode",
    "AperiodicFragment",
    "AperiodicServiceAdaptor",
    "AsyncDataLinkParams",
    "AsyncFlowControl",
    "EventGroupSet",
    "PeriodicServiceAdaptor",
    "SyncDataDiscard",
    "SyncDataLinkParams",
    "SyncFlowControl",
    "TransmissionMode",
]


import math
from dataclasses import dataclass
from enum import IntEnum

# ---------------------------------------------------------------------------
# 6.5 传输类型枚举
# ---------------------------------------------------------------------------


class TransmissionMode(IntEnum):
    """传输模式 (6.5.1.4 ~ 6.5.1.7, 6.5.2.6 ~ 6.5.2.10)。"""
    UNICAST = 0
    MULTICAST = 1
    BIDIRECTIONAL_MULTICAST = 2
    FEEDBACK_MULTICAST = 3
    BROADCAST_ASYNC = 4
    BROADCAST_SYNC = 5


class AdaptMode(IntEnum):
    """同步业务适配方式 (6.5.3.1)。"""
    PERIODIC = 0
    APERIODIC = 1


# ---------------------------------------------------------------------------
# 6.5.1.2 异步数据链路参数
# ---------------------------------------------------------------------------


@dataclass
class AsyncDataLinkParams:
    """异步数据链路参数 (6.5.1.2)。"""
    event_group_start_us: int = 0
    event_group_period_us: int = 0
    event_period_us: int = 0
    intra_event_interval_us: int = 0
    inter_event_interval_us: int = 0
    event_group_interval_us: int = 0
    first_tx: bool = True
    tx_pdu_max: int = 251
    rx_pdu_max: int = 251
    tx_max_time_us: int = 0
    rx_max_time_us: int = 0
    tx_time_offset: int = 0
    rx_time_offset: int = 0
    tx_max_time_offset: int = 0
    rx_max_time_offset: int = 0
    tx_sdu_max: int = 0
    rx_sdu_max: int = 0
    mode: TransmissionMode = TransmissionMode.UNICAST

    @property
    def max_event_length(self) -> int:
        """事件最大长度 = 先发最大偏移 + 后发最大偏移。"""
        return self.tx_max_time_offset + self.rx_max_time_offset


# ---------------------------------------------------------------------------
# 6.5.1.3 异步数据链路流控
# ---------------------------------------------------------------------------


class AsyncFlowControl:
    """异步数据链路流控决策 (6.5.1.3)。

    根据标准规则, 判断先发节点是否应停止在当前事件组
    后续事件中发送。
    """

    def should_stop_unicast_tx(
        self,
        tx_flow_ctrl: int,
        rx_flow_ctrl: int | None,
        rx_ack: bool | None,
        rx_data_len: int | None,
    ) -> bool:
        """单播先发节点停止条件 (6.5.1.3)。

        条件一: 先发发送 flow_ctrl=0 且收到后发 flow_ctrl=0, ACK, data_len=0
        条件二: 先发发送了消息但未收到后发消息 (rx 参数为 None)
        """
        if rx_flow_ctrl is None or rx_ack is None or rx_data_len is None:
            return True
        return tx_flow_ctrl == 0 and rx_flow_ctrl == 0 and rx_ack and rx_data_len == 0

    def should_stop_multicast_semi_reliable(
        self, tx_flow_ctrl: int, any_nack: bool,
    ) -> bool:
        """半可靠组播组长停止条件。

        组长发送 flow=0 且未收到任何 NACK。
        """
        return tx_flow_ctrl == 0 and not any_nack

    def should_stop_multicast_full(
        self,
        tx_flow_ctrl: int,
        member_feedbacks: list[tuple[int, bool, int]],
    ) -> bool:
        """非半可靠组播/双向组播组长停止条件。

        组长发送 flow=0, 收到所有组员 flow=0、ACK、data_len=0。
        """
        if tx_flow_ctrl != 0:
            return False
        for flow, ack, data_len in member_feedbacks:
            if flow != 0 or not ack or data_len != 0:
                return False
        return True

    def should_stop_feedback_multicast_leader(
        self,
        tx_flow_ctrl: int,
        member_feedbacks: list[tuple[int, bool, int]],
    ) -> bool:
        """反馈组播组长停止条件, 同 should_stop_multicast_full。"""
        return self.should_stop_multicast_full(tx_flow_ctrl, member_feedbacks)

    def should_stop_feedback_multicast_member(
        self, prev_flow_ctrl: int, leader_ack: bool,
    ) -> bool:
        """反馈组播组员停止条件。

        前一事件中组员发送 flow=0, 当前事件收到组长 ACK=成功。
        """
        return prev_flow_ctrl == 0 and leader_ack


# ---------------------------------------------------------------------------
# 6.5.2.2 同步数据链路参数
# ---------------------------------------------------------------------------


@dataclass
class SyncDataLinkParams:
    """同步数据链路参数 (6.5.2.2)。"""
    event_group_start_us: int = 0
    event_group_period_us: int = 0
    event_period_us: int = 0
    intra_event_interval_us: int = 0
    inter_event_interval_us: int = 0
    event_group_interval_us: int = 0
    event_count: int = 1
    first_tx: bool = True
    tx_pdu_max: int = 251
    rx_pdu_max: int = 251
    tx_max_time_us: int = 0
    rx_max_time_us: int = 0
    tx_time_offset: int = 0
    rx_time_offset: int = 0
    tx_max_time_offset: int = 0
    rx_max_time_offset: int = 0
    tx_sdu_max: int = 0
    rx_sdu_max: int = 0
    new_pkt_count: int = 1
    discard_period: int = 3
    adapt_mode: AdaptMode = AdaptMode.PERIODIC
    sync_anchor_delay_us: int = 0
    sync_ref_delay_us: int = 0
    mode: TransmissionMode = TransmissionMode.UNICAST

    @property
    def max_event_length(self) -> int:
        return self.tx_max_time_offset + self.rx_max_time_offset


# ---------------------------------------------------------------------------
# 6.5.2.4 同步数据链路数据丢弃机制
# ---------------------------------------------------------------------------


@dataclass
class SyncDataDiscard:
    """同步数据丢弃机制 (6.5.2.4)。

    跟踪发送序列号、本地基准值和有效载荷计数, 支持
    主动丢弃和被动丢弃判定。
    """
    new_pkt_count: int = 2
    discard_period: int = 3
    tx_sn: int = 0
    event_group_index: int = 0
    payload_count: int = 0
    _last_ack: bool | None = None

    @property
    def local_baseline(self) -> int:
        """本地基准值 (6.5.2.4)。"""
        if self.event_group_index < self.discard_period:
            return 0
        return self.new_pkt_count * (
            self.event_group_index - self.discard_period + 1
        )

    def on_ack(self) -> None:
        """收到 ACK, 发送序列号 +1。"""
        self._last_ack = True
        self.tx_sn += 1
        self.payload_count += 1

    def on_nack_or_timeout(self) -> None:
        """收到 NACK 或未收到反馈, 序列号不变。"""
        self._last_ack = False

    def on_event_group_boundary(self) -> None:
        """跨事件组边界, 更新序列号和事件组索引。"""
        self.tx_sn = max(0, self.tx_sn - self.new_pkt_count)
        self.event_group_index += 1

    def is_discarded(self, payload_id: int) -> bool:
        """判定 payload_id 是否已被被动丢弃。"""
        return payload_id < self.local_baseline

    def should_active_discard(self, max_retransmit: int) -> bool:
        """主动丢弃判定: 对于 FT3/FT4, 发送端可自主决定。"""
        return self._last_ack is False and max_retransmit > 0

    def active_discard(self) -> None:
        """执行主动丢弃, 翻转序列号发送下一包。"""
        self.tx_sn += 1
        self.payload_count += 1


# ---------------------------------------------------------------------------
# 6.5.2.3 同步数据链路流控
# ---------------------------------------------------------------------------


class SyncFlowControl:
    """同步数据链路流控 (6.5.2.3)。

    与异步流控逻辑基本一致, 但同步链路中先发节点在
    满足停止条件后直接停止, 后发节点的行为由事件组
    内事件总数约束。
    """

    def should_stop_unicast_tx(
        self,
        tx_flow_ctrl: int,
        rx_flow_ctrl: int | None,
        rx_ack: bool | None,
        rx_data_len: int | None,
    ) -> bool:
        """单播先发停止 (6.5.2.3)。"""
        if rx_flow_ctrl is None or rx_ack is None or rx_data_len is None:
            return False
        return (
            tx_flow_ctrl == 0
            and rx_flow_ctrl == 0
            and rx_ack
            and rx_data_len == 0
        )

    def should_stop_multicast_semi_reliable(
        self, tx_flow_ctrl: int, any_nack: bool,
    ) -> bool:
        return tx_flow_ctrl == 0 and not any_nack

    def should_stop_multicast_full(
        self,
        tx_flow_ctrl: int,
        member_feedbacks: list[tuple[int, bool, int]],
    ) -> bool:
        if tx_flow_ctrl != 0:
            return False
        for flow, ack, data_len in member_feedbacks:
            if flow != 0 or not ack or data_len != 0:
                return False
        return True

    def should_stop_feedback_leader(
        self,
        tx_flow_ctrl: int,
        member_feedbacks: list[tuple[int, bool, int]],
    ) -> bool:
        return self.should_stop_multicast_full(tx_flow_ctrl, member_feedbacks)

    def should_stop_feedback_member(
        self, prev_flow_ctrl: int, leader_ack: bool,
    ) -> bool:
        return prev_flow_ctrl == 0 and leader_ack


# ---------------------------------------------------------------------------
# 6.5.2.5 事件组集合
# ---------------------------------------------------------------------------


@dataclass
class EventGroupSet:
    """事件组集合 (6.5.2.5)。

    多个具有相同事件组周期的事件组可组合为事件组集合。
    """
    event_group_count: int = 1
    event_group_spacing_slots: int = 0
    event_group_period_us: int = 0
    schedule_slot_us: int = 125

    def event_group_offsets(self) -> list[int]:
        """返回集合中各事件组相对于集合起始的偏移 (μs)。"""
        return [
            i * self.event_group_spacing_slots * self.schedule_slot_us
            for i in range(self.event_group_count)
        ]

    def set_duration_us(self) -> int:
        """集合总时长 (μs): 最后一个事件组起始 + 事件组周期。"""
        offsets = self.event_group_offsets()
        if not offsets:
            return 0
        return offsets[-1] + self.event_group_period_us

    def validate(self) -> bool:
        """校验事件组集合的时间资源不能交叠。"""
        offsets = self.event_group_offsets()
        for i in range(len(offsets) - 1):
            if offsets[i] + self.event_group_period_us > offsets[i + 1]:
                return False
        return True


# ---------------------------------------------------------------------------
# 6.5.3.2 周期适配方法
# ---------------------------------------------------------------------------


@dataclass
class PeriodicServiceAdaptor:
    """同步数据链路周期适配 (6.5.3.2)。

    将上层 SDU 分段为 PDU, 并按事件组周期进行时间同步。
    """
    sdu_max: int = 256
    pdu_max: int = 251
    event_group_period_us: int = 0
    sdu_period_us: int = 0
    sync_ref_delay_us: int = 0
    sync_anchor_delay_us: int = 0
    discard_period: int = 3

    @property
    def segments_per_sdu(self) -> int:
        """每个 SDU 需要的 PDU 分段数。"""
        if self.pdu_max <= 0:
            return 1
        return math.ceil(self.sdu_max / self.pdu_max)

    @property
    def sdus_per_event_group(self) -> int:
        """每个事件组周期内传输的 SDU 数量。"""
        if self.sdu_period_us <= 0:
            return 1
        return self.event_group_period_us // self.sdu_period_us

    @property
    def frames_per_event_group(self) -> int:
        """每个事件组周期内传输的无线帧数。"""
        return self.segments_per_sdu * self.sdus_per_event_group

    def segment_sdu(self, sdu: bytes) -> list[bytes]:
        """将 SDU 分段为 PDU 列表。"""
        if not sdu:
            return [b""]
        segments: list[bytes] = [
            sdu[offset: offset + self.pdu_max]
            for offset in range(0, len(sdu), self.pdu_max)
        ]
        expected = self.segments_per_sdu
        while len(segments) < expected:
            segments.append(b"")
        return segments

    def reassemble_sdu(self, pdus: list[bytes]) -> bytes:
        """将 PDU 列表重组为 SDU。"""
        return b"".join(pdus)

    def tx_accept_time(
        self, event_group_start_us: int,
    ) -> int:
        """发送端 t0: 开始接收 SDU 的时刻。"""
        sdus_n = self.sdus_per_event_group
        t1 = event_group_start_us
        return t1 - (sdus_n - 1) * self.sdu_period_us - self.sync_ref_delay_us

    def rx_deliver_time(
        self, event_group_start_us: int,
    ) -> int:
        """接收端 r1: 向高层递交第一个 SDU 的时刻。"""
        return (
            event_group_start_us
            + self.sync_anchor_delay_us
            + (self.discard_period - 1) * self.event_group_period_us
        )


# ---------------------------------------------------------------------------
# 6.5.3.3 非周期适配方法
# ---------------------------------------------------------------------------


@dataclass
class AperiodicFragment:
    """非周期适配分片 (6.5.3.3)。"""
    is_first: bool = False
    is_last: bool = False
    time_offset_us: int = 0
    data: bytes = b""


class AperiodicServiceAdaptor:
    """同步数据链路非周期适配 (6.5.3.3)。

    将 SDU 分片, 每个包含首段的分片携带时间偏移量。
    """

    def __init__(self, pdu_max: int = 251, header_size: int = 4) -> None:
        self.pdu_max = pdu_max
        self.header_size = header_size

    def fragment_sdu(
        self, sdu: bytes, time_offset_us: int = 0,
    ) -> list[AperiodicFragment]:
        """将 SDU 分片。"""
        payload_cap = self.pdu_max - self.header_size
        if payload_cap <= 0:
            return [AperiodicFragment(
                is_first=True, is_last=True,
                time_offset_us=time_offset_us, data=sdu,
            )]
        fragments: list[AperiodicFragment] = []
        offset = 0
        while offset < len(sdu):
            chunk = sdu[offset: offset + payload_cap]
            frag = AperiodicFragment(
                is_first=(offset == 0),
                is_last=(offset + payload_cap >= len(sdu)),
                time_offset_us=time_offset_us if offset == 0 else 0,
                data=chunk,
            )
            fragments.append(frag)
            offset += payload_cap
        if not fragments:
            fragments.append(AperiodicFragment(
                is_first=True, is_last=True,
                time_offset_us=time_offset_us, data=b"",
            ))
        return fragments

    def reassemble_sdu(
        self, fragments: list[AperiodicFragment],
    ) -> tuple[bytes, int]:
        """从分片重组 SDU, 返回 (sdu, time_offset_us)。"""
        time_offset = 0
        data_parts: list[bytes] = []
        for frag in fragments:
            if frag.is_first:
                time_offset = frag.time_offset_us
            data_parts.append(frag.data)
        return b"".join(data_parts), time_offset

    def rx_deliver_time(
        self,
        event_group_start_us: int,
        sync_anchor_delay_us: int,
        discard_period: int,
        event_group_period_us: int,
        sdu_period_us: int,
        sdu_time_offset_us: int,
    ) -> int:
        """接收端 r1 (6.5.3.3.2)。"""
        return (
            event_group_start_us
            + sync_anchor_delay_us
            + discard_period * event_group_period_us
            + sdu_period_us
            - sdu_time_offset_us
        )
