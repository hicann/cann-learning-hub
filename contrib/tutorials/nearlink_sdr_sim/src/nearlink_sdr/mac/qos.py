"""MAC 层服务质量管理 -- TXS-10002-2025 标准 6.5/7.2

实现异步/同步数据链路的数据传输质量控制:
- ARQ 自动重传请求: 序列号管理、ACK/NACK 反馈、重传决策 (6.5.1)
- HARQ 混合自动重传: 基于 TB 和 CBG 的反馈模式 (6.5.1.1)
- 流控: 发送缓冲区管理与流控信令 (控制信息 flow_ctrl 字段)
- 链路质量跟踪: MCS 反馈、FER 估计、AMC 建议 (控制信息 LQI 字段)
- 发送队列: 优先级分级与 PDU 管理
"""

from __future__ import annotations

__all__ = [
    "MAX_CBG",
    "MAX_SN_FT1",
    "MAX_SN_FT2",
    "MAX_SN_FT34",
    "ArqState",
    "FeedbackMode",
    "FlowController",
    "HarqController",
    "LinkQualityTracker",
    "LinkType",
    "Priority",
    "QosManager",
    "TxDecision",
    "TxQueue",
    "TxQueueItem",
]


from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_SN_FT1 = 2        # FT1: 1-bit SN, 取值 0/1
MAX_SN_FT2 = 2        # FT2 异步模式 A2: 1-bit SN
MAX_SN_FT34 = 32      # FT3/FT4 异步模式 A3: 5-bit SN
MAX_CBG = 8            # 最大编码块组数


class FeedbackMode(IntEnum):
    """HARQ 反馈模式。"""
    TB = 0   # 基于传输块整体
    CBG = 1  # 基于编码块组


class LinkType(IntEnum):
    """数据链路类型。"""
    ASYNC = 0   # 异步数据链路 (可靠传输)
    SYNC = 1    # 同步数据链路 (非可靠传输)


class TxDecision(IntEnum):
    """发送决策。"""
    NEW_DATA = auto()       # 发送新数据
    RETRANSMIT = auto()     # 重传旧数据
    RETRANSMIT_CBG = auto() # 部分 CBG 重传 (MCS=15)
    EMPTY = auto()          # 发送空包


class Priority(IntEnum):
    """数据传输优先级 (数值越小优先级越高)。"""
    CONTROL = 0    # 控制面信令
    REALTIME = 1   # 实时数据 (同步等时)
    HIGH = 2       # 高优先级数据
    NORMAL = 3     # 常规数据
    LOW = 4        # 低优先级/后台数据


# ---------------------------------------------------------------------------
# 6.5.1 ARQ 控制器
# ---------------------------------------------------------------------------


@dataclass
class ArqState:
    """ARQ 序列号与重传状态。

    标准 6.5.1.1 描述的异步数据链路传输流程:
    - tx_sn: 当前发送序列号
    - expected_rx_sn: 期望接收的对端序列号
    - pending_ack: 当前等待 ACK 的数据是否未确认
    - retransmit_count: 当前包的重传次数
    - max_retransmit: 最大重传次数 (0 = 无限, 异步链路默认)
    """
    frame_type: int = 2
    link_type: LinkType = LinkType.ASYNC
    tx_sn: int = 0
    expected_rx_sn: int = 0
    pending_ack: bool = False
    retransmit_count: int = 0
    max_retransmit: int = 0
    _last_tx_data: bytes = b""

    @property
    def sn_modulus(self) -> int:
        if self.frame_type == 1:
            return MAX_SN_FT1
        if self.frame_type == 2:
            return MAX_SN_FT2
        return MAX_SN_FT34

    def on_tx_new(self, data: bytes) -> int:
        """发送新数据包, 返回使用的 SN。"""
        self._last_tx_data = data
        self.pending_ack = True
        self.retransmit_count = 0
        return self.tx_sn

    def on_ack_received(self) -> None:
        """收到 ACK: 翻转 SN, 清除重传状态。"""
        self.pending_ack = False
        self.retransmit_count = 0
        self.tx_sn = (self.tx_sn + 1) % self.sn_modulus

    def on_nack_received(self) -> TxDecision:
        """收到 NACK: 决定是否重传。

        异步链路: 持续重传直到收到 ACK。
        同步链路: 超出 max_retransmit 后丢弃。
        """
        if (
            self.link_type == LinkType.SYNC
            and self.max_retransmit > 0
            and self.retransmit_count >= self.max_retransmit
        ):
            self._discard_pending()
            return TxDecision.NEW_DATA
        self.retransmit_count += 1
        return TxDecision.RETRANSMIT

    def on_no_feedback(self) -> TxDecision:
        """未收到反馈: 与 NACK 处理相同。"""
        return self.on_nack_received()

    def on_rx_packet(self, rx_sn: int) -> bool:
        """接收数据包, 返回是否为新数据。

        若 rx_sn == expected_rx_sn: 新数据, 反馈 ACK。
        否则: 重复包, 仍反馈 ACK。
        """
        if rx_sn == self.expected_rx_sn:
            self.expected_rx_sn = (self.expected_rx_sn + 1) % self.sn_modulus
            return True
        return False

    def _discard_pending(self) -> None:
        """丢弃当前待确认数据 (同步链路超时)。"""
        self.pending_ack = False
        self.retransmit_count = 0
        self._last_tx_data = b""
        self.tx_sn = (self.tx_sn + 1) % self.sn_modulus


# ---------------------------------------------------------------------------
# 6.5.1.1 HARQ 控制器
# ---------------------------------------------------------------------------


@dataclass
class HarqController:
    """混合自动重传请求控制器。

    管理 HARQ 反馈字段 (8 bit) 和 CBG 级部分重传逻辑。

    TB 模式: harq_feedback 指示期望对端传输的数据包 SN。
    CBG 模式: harq_feedback 每 bit 对应一个 CBG 的 ACK/NACK。
    """
    feedback_mode: FeedbackMode = FeedbackMode.TB
    n_cbg: int = 1
    _cbg_status: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._cbg_status:
            self._cbg_status = [False] * min(self.n_cbg, MAX_CBG)

    def encode_harq_feedback(self, ack: bool, peer_sn: int = 0) -> int:
        """编码 HARQ 反馈字段 (8 bit)。

        TB 模式: 返回期望对端传输的 SN。
        CBG 模式: 返回 CBG ACK/NACK bitmap。
        """
        if self.feedback_mode == FeedbackMode.TB:
            return peer_sn & 0xFF
        # CBG 模式: 每 bit 对应一个 CBG
        bitmap = 0
        for i, ok in enumerate(self._cbg_status):
            if ok:
                bitmap |= (1 << i)
        return bitmap & 0xFF

    def decode_harq_feedback(self, harq_field: int) -> list[bool]:
        """解码 HARQ 反馈字段。

        TB 模式: 返回 [全部 ACK] 或 [全部 NACK]。
        CBG 模式: 返回每个 CBG 的 ACK 状态。
        """
        if self.feedback_mode == FeedbackMode.TB:
            return [True] * self.n_cbg
        return [(harq_field >> i) & 1 == 1 for i in range(self.n_cbg)]

    def update_cbg_status(self, cbg_ack: list[bool]) -> None:
        """更新 CBG 接收状态。"""
        for i in range(min(len(cbg_ack), len(self._cbg_status))):
            if cbg_ack[i]:
                self._cbg_status[i] = True

    def needs_cbg_retransmit(self) -> bool:
        """是否需要 CBG 级部分重传。"""
        if self.feedback_mode != FeedbackMode.CBG:
            return False
        return any(self._cbg_status) and not all(self._cbg_status)

    def get_retransmit_cbg_mask(self) -> int:
        """获取需要重传的 CBG 掩码 (用于 B1 控制信息 data_length 字段)。

        标准 6.5.1.1: MCS=15 时, data_length 低 M bit 指示需要传输的 CBG。
        """
        mask = 0
        for i, ok in enumerate(self._cbg_status):
            if not ok:
                mask |= (1 << i)
        return mask

    def decide_tx(self, ack_received: bool, arq: ArqState) -> TxDecision:
        """综合 HARQ 反馈决定发送行为。"""
        if ack_received:
            arq.on_ack_received()
            self.reset_cbg()
            return TxDecision.NEW_DATA

        if self.needs_cbg_retransmit():
            arq.retransmit_count += 1
            return TxDecision.RETRANSMIT_CBG

        return arq.on_nack_received()

    def reset_cbg(self) -> None:
        """重置 CBG 状态 (新数据传输前)。"""
        self._cbg_status = [False] * len(self._cbg_status)

    @property
    def all_cbg_acked(self) -> bool:
        return all(self._cbg_status)


# ---------------------------------------------------------------------------
# 流控管理
# ---------------------------------------------------------------------------


@dataclass
class FlowController:
    """发送流控管理器。

    根据发送缓冲区状态生成 flow_ctrl 指示字段:
    - 0: 没有后续数据
    - 1: 有后续数据待发送

    同时实现简单的背压机制。
    """
    buffer_high_watermark: int = 16
    buffer_low_watermark: int = 4
    _buffer_count: int = 0
    _paused: bool = False

    def enqueue(self, count: int = 1) -> None:
        """数据入队。"""
        self._buffer_count += count
        if self._buffer_count >= self.buffer_high_watermark:
            self._paused = True

    def dequeue(self, count: int = 1) -> None:
        """数据出队。"""
        self._buffer_count = max(0, self._buffer_count - count)
        if self._buffer_count <= self.buffer_low_watermark:
            self._paused = False

    @property
    def flow_ctrl_bit(self) -> int:
        """控制信息 flow_ctrl 字段值。"""
        return 1 if self._buffer_count > 0 else 0

    @property
    def is_paused(self) -> bool:
        """是否处于背压暂停状态。"""
        return self._paused

    @property
    def buffer_count(self) -> int:
        return self._buffer_count


# ---------------------------------------------------------------------------
# 链路质量跟踪
# ---------------------------------------------------------------------------


@dataclass
class LinkQualityTracker:
    """链路质量跟踪与 AMC 建议。

    跟踪最近 N 帧的 CRC 通过率, 计算瞬时 FER,
    根据 FER 阈值建议 MCS 调整方向。

    标准: 控制信息 LQI 字段 (前 4 bit 为 MCS 指示)。
    """
    window_size: int = 32
    fer_target_low: float = 0.01
    fer_target_high: float = 0.10
    _history: deque[bool] = field(default_factory=deque)
    _current_mcs: int = 7

    @property
    def current_mcs(self) -> int:
        return self._current_mcs

    @current_mcs.setter
    def current_mcs(self, value: int) -> None:
        self._current_mcs = max(0, min(12, value))

    def record(self, crc_ok: bool) -> None:
        """记录一帧接收结果。"""
        self._history.append(crc_ok)
        if len(self._history) > self.window_size:
            self._history.popleft()

    @property
    def fer(self) -> float:
        """当前窗口 FER。"""
        if not self._history:
            return 0.0
        n_fail = sum(1 for ok in self._history if not ok)
        return n_fail / len(self._history)

    @property
    def success_rate(self) -> float:
        """当前窗口成功率。"""
        return 1.0 - self.fer

    def suggest_mcs_adjustment(self) -> int:
        """建议 MCS 调整方向。

        :returns: +1: 建议提升 MCS (链路质量好)
            -1: 建议降低 MCS (链路质量差)
             0: 保持当前 MCS
        """
        if len(self._history) < self.window_size // 2:
            return 0
        if self.fer < self.fer_target_low and self._current_mcs < 12:
            return 1
        if self.fer > self.fer_target_high and self._current_mcs > 0:
            return -1
        return 0

    def apply_suggestion(self) -> int:
        """应用 MCS 调整建议并返回新 MCS。"""
        adj = self.suggest_mcs_adjustment()
        self.current_mcs = self._current_mcs + adj
        return self._current_mcs

    def encode_lqi(self) -> int:
        """编码 LQI 字段 (8 bit): 前 4 bit MCS, 后 4 bit 预留。"""
        return (self._current_mcs & 0x0F) << 4

    @staticmethod
    def decode_lqi(lqi: int) -> int:
        """解码 LQI 字段, 提取 MCS 指示。"""
        return (lqi >> 4) & 0x0F


# ---------------------------------------------------------------------------
# 发送队列
# ---------------------------------------------------------------------------


@dataclass
class TxQueueItem:
    """发送队列项。"""
    priority: Priority
    data: bytes
    segment_type: int = 0
    retransmit: bool = False
    attempt: int = 0


class TxQueue:
    """优先级发送队列。

    高优先级数据优先出队; 同优先级内按FIFO顺序;
    重传数据优先于同优先级新数据。
    """

    def __init__(self, max_size: int = 64) -> None:
        self._queues: dict[Priority, deque[TxQueueItem]] = {
            p: deque() for p in Priority
        }
        self._max_size = max_size
        self._total = 0

    def push(self, item: TxQueueItem) -> bool:
        """入队, 队满时返回 False。"""
        if self._total >= self._max_size:
            return False
        q = self._queues[item.priority]
        if item.retransmit:
            q.appendleft(item)
        else:
            q.append(item)
        self._total += 1
        return True

    def pop(self) -> TxQueueItem | None:
        """出队最高优先级项。"""
        for p in Priority:
            q = self._queues[p]
            if q:
                self._total -= 1
                return q.popleft()
        return None

    def peek(self) -> TxQueueItem | None:
        """查看但不移除最高优先级项。"""
        for p in Priority:
            q = self._queues[p]
            if q:
                return q[0]
        return None

    @property
    def size(self) -> int:
        return self._total

    @property
    def is_empty(self) -> bool:
        return self._total == 0

    @property
    def is_full(self) -> bool:
        return self._total >= self._max_size

    def count_by_priority(self, priority: Priority) -> int:
        """指定优先级的队列项数量。"""
        return len(self._queues[priority])

    def clear(self) -> None:
        """清空所有队列。"""
        for q in self._queues.values():
            q.clear()
        self._total = 0


# ---------------------------------------------------------------------------
# QoS 管理器 (集成)
# ---------------------------------------------------------------------------


@dataclass
class QosManager:
    """QoS 服务质量管理器。

    集成 ARQ、HARQ、流控、链路质量跟踪和发送队列,
    为 MAC-PHY 接口提供统一的数据传输质量管理。
    """
    arq: ArqState = field(default_factory=ArqState)
    harq: HarqController = field(default_factory=HarqController)
    flow: FlowController = field(default_factory=FlowController)
    quality: LinkQualityTracker = field(default_factory=LinkQualityTracker)
    tx_queue: TxQueue = field(default_factory=TxQueue)

    def prepare_tx(self) -> tuple[TxDecision, TxQueueItem | None]:
        """准备下一帧发送。

        :returns: (发送决策, 队列项或 None)
        """
        if self.arq.pending_ack:
            return TxDecision.RETRANSMIT, None

        item = self.tx_queue.pop()
        if item is None:
            return TxDecision.EMPTY, None

        self.arq.on_tx_new(item.data)
        self.flow.dequeue()
        return TxDecision.NEW_DATA, item

    def on_tx_feedback(self, crc_ok: bool) -> TxDecision:
        """处理来自对端的 ACK/NACK 反馈。"""
        self.quality.record(crc_ok)
        if crc_ok:
            return self.harq.decide_tx(True, self.arq)
        return self.harq.decide_tx(False, self.arq)

    def submit_data(
        self,
        data: bytes,
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """提交数据到发送队列。"""
        item = TxQueueItem(priority=priority, data=data)
        ok = self.tx_queue.push(item)
        if ok:
            self.flow.enqueue()
        return ok

    def get_ctrl_fields(self) -> dict[str, int]:
        """获取当前控制信息字段值。"""
        return {
            "tx_sn": self.arq.tx_sn,
            "rx_sn": self.arq.expected_rx_sn,
            "flow_ctrl": self.flow.flow_ctrl_bit,
            "empty_packet": 1 if self.tx_queue.is_empty and not self.arq.pending_ack else 0,
            "harq_feedback": self.harq.encode_harq_feedback(
                not self.arq.pending_ack, self.arq.tx_sn,
            ),
        }
