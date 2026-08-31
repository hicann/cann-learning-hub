"""链路管理状态机 -- TXS-10002-2025 标准 7.1 / 7.2。

实现 SparkLink SLE 的链路生命周期管理，包括广播、发现、接入、
链接态控制面流程和链路断开。
"""

from __future__ import annotations

__all__ = [
    "DisconnectReason",
    "Event",
    "EventType",
    "InvalidState",
    "InvalidTransition",
    "LinkManager",
    "LinkManagerCallback",
    "LinkParams",
    "LinkState",
    "Role",
]


import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from nearlink_sdr.mac.broadcast import BroadcastFrame
from nearlink_sdr.mac.frame import ControlFrame
from nearlink_sdr.mac.signaling import decode_signaling, encode_signaling

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# 角色与状态枚举
# -----------------------------------------------------------------------

class Role(Enum):
    """G/T 节点角色（标准 7.1.3 / 7.1.7.1）。"""
    NONE = auto()
    G_NODE = auto()
    T_NODE = auto()


class LinkState(Enum):
    """链路状态。

    IDLE        -- 初始待机
    BROADCASTING -- 广播态，周期发送基础/扩展广播帧（7.1.1）
    SCANNING    -- 发现态，接收广播帧（7.1.2）
    ACCESSING   -- 接入态，发送接入请求/等待响应（7.1.3）
    CONNECTED   -- 链接态，异步数据链路已建立
    PAIRING     -- 配对态，执行安全流程（9.2）
    DORMANT     -- 休眠态，保持链路但停止数据传输（7.2.14）
    DISCONNECTED -- 链路已断开
    """
    IDLE = auto()
    BROADCASTING = auto()
    SCANNING = auto()
    ACCESSING = auto()
    CONNECTED = auto()
    PAIRING = auto()
    DORMANT = auto()
    DISCONNECTED = auto()


class DisconnectReason(Enum):
    """断开原因。"""
    LOCAL_REQUEST = 0
    REMOTE_REQUEST = 1
    TIMEOUT = 2
    ACCESS_REJECTED = 3


# -----------------------------------------------------------------------
# 链路参数
# -----------------------------------------------------------------------

@dataclass
class LinkParams:
    """链路运行参数集合。"""
    # 收发间隔 (7.2.1)
    tx_rx_interval: int = 20  # 单位 ms
    # 数据长度 (7.2.6)
    tx_max_octets: int = 27
    rx_max_octets: int = 27
    # 跳频信道表
    channel_table: bytes = b""
    # 帧类型
    frame_type: int = 2
    # 带宽
    bandwidth: int = 0
    # CRC 类型
    crc_type: int = 0
    # 超时 (单位 ms)
    supervision_timeout: int = 5000


# -----------------------------------------------------------------------
# 事件
# -----------------------------------------------------------------------

class EventType(Enum):
    """状态机输入事件类型。"""
    # 广播/发现/接入
    START_BROADCAST = auto()
    STOP_BROADCAST = auto()
    START_SCAN = auto()
    STOP_SCAN = auto()
    BROADCAST_RECEIVED = auto()
    SEND_ACCESS_REQUEST = auto()
    ACCESS_RESPONSE_RECEIVED = auto()
    ACCESS_REQUEST_RECEIVED = auto()
    # 链接态控制面 (7.2)
    SIGNALING_RECEIVED = auto()
    SIGNALING_SEND = auto()
    # 配对（9.2）
    START_PAIRING = auto()
    PAIRING_COMPLETE = auto()
    PAIRING_FAILED = auto()
    # 断开
    DISCONNECT_REQUEST = auto()
    DISCONNECT_RECEIVED = auto()
    SUPERVISION_TIMEOUT = auto()
    # 休眠/唤醒（7.2.14）
    SLEEP_REQUEST = auto()
    WAKE_REQUEST = auto()
    WAKE_RECEIVED = auto()


@dataclass
class Event:
    """状态机事件。"""
    type: EventType
    data: Any = None


# -----------------------------------------------------------------------
# 回调协议
# -----------------------------------------------------------------------

class LinkManagerCallback:
    """链路管理回调接口。子类可覆盖需要的方法。"""

    def on_state_changed(self, old: LinkState, new: LinkState) -> None:
        """状态变迁通知。"""

    def on_broadcast_frame(self, frame: BroadcastFrame) -> None:
        """接收到广播帧。"""

    def on_access_accepted(self, role: Role) -> None:
        """接入成功，角色已确定。"""

    def on_access_rejected(self) -> None:
        """接入被拒绝。"""

    def on_signaling(self, msg: Any) -> None:
        """收到控制面信令。"""

    def on_disconnected(self, reason: DisconnectReason) -> None:
        """链路断开。"""

    def on_dormant(self) -> None:
        """进入休眠态（7.2.14）。"""

    def on_wakeup(self) -> None:
        """从休眠态唤醒（7.2.14）。"""


# -----------------------------------------------------------------------
# 链路管理器
# -----------------------------------------------------------------------

_VALID_TRANSITIONS: dict[LinkState, set[LinkState]] = {
    LinkState.IDLE: {
        LinkState.BROADCASTING,
        LinkState.SCANNING,
    },
    LinkState.BROADCASTING: {
        LinkState.IDLE,
        LinkState.CONNECTED,
    },
    LinkState.SCANNING: {
        LinkState.IDLE,
        LinkState.ACCESSING,
    },
    LinkState.ACCESSING: {
        LinkState.CONNECTED,
        LinkState.SCANNING,
        LinkState.DISCONNECTED,
    },
    LinkState.CONNECTED: {
        LinkState.PAIRING,
        LinkState.DORMANT,
        LinkState.DISCONNECTED,
    },
    LinkState.PAIRING: {
        LinkState.CONNECTED,
        LinkState.DISCONNECTED,
    },
    LinkState.DORMANT: {
        LinkState.CONNECTED,
        LinkState.DISCONNECTED,
    },
    LinkState.DISCONNECTED: {
        LinkState.IDLE,
    },
}


@dataclass
class LinkManager:
    """SparkLink SLE 链路管理状态机。

    以事件驱动方式管理链路生命周期：
    IDLE → BROADCASTING / SCANNING → ACCESSING → CONNECTED → DISCONNECTED。
    """
    state: LinkState = LinkState.IDLE
    role: Role = Role.NONE
    params: LinkParams = field(default_factory=LinkParams)
    callback: LinkManagerCallback = field(default_factory=LinkManagerCallback)
    peer_address: bytes = b""
    local_address: bytes = b""
    # 内部计数
    _slot_counter: int = 0
    _last_rx_time: float = 0.0
    _event_log: list[tuple[float, EventType, LinkState, LinkState]] = field(
        default_factory=list
    )

    # ---------------------------------------------------------------
    # 公共接口
    # ---------------------------------------------------------------

    def process_event(self, event: Event) -> None:
        """处理一个事件，驱动状态转换。"""
        handler = _HANDLERS.get((self.state, event.type))
        if handler is None:
            log.warning(
                "忽略事件 %s (当前状态 %s)", event.type.name, self.state.name
            )
            return
        handler(self, event)

    def reset(self) -> None:
        """重置到 IDLE 状态。"""
        self._transition(LinkState.IDLE)
        self.role = Role.NONE
        self.peer_address = b""

    @property
    def is_connected(self) -> bool:
        return self.state == LinkState.CONNECTED

    @property
    def is_dormant(self) -> bool:
        return self.state == LinkState.DORMANT

    @property
    def event_log(self) -> list[tuple[float, EventType, LinkState, LinkState]]:
        return list(self._event_log)

    # ---------------------------------------------------------------
    # 状态转换
    # ---------------------------------------------------------------

    def _transition(self, new_state: LinkState) -> None:
        old = self.state
        if new_state not in _VALID_TRANSITIONS.get(old, set()):
            if old == new_state:
                return
            raise InvalidTransition(
                f"非法转换: {old.name} → {new_state.name}"
            )
        self.state = new_state
        ts = time.monotonic()
        self._event_log.append((ts, EventType.START_BROADCAST, old, new_state))
        log.info("状态转换 %s → %s", old.name, new_state.name)
        self.callback.on_state_changed(old, new_state)

    # ---------------------------------------------------------------
    # 事件处理器
    # ---------------------------------------------------------------

    def _handle_start_broadcast(self, event: Event) -> None:
        self._transition(LinkState.BROADCASTING)

    def _handle_stop_broadcast(self, event: Event) -> None:
        self._transition(LinkState.IDLE)

    def _handle_start_scan(self, event: Event) -> None:
        self._transition(LinkState.SCANNING)

    def _handle_stop_scan(self, event: Event) -> None:
        self._transition(LinkState.IDLE)

    def _handle_broadcast_received(self, event: Event) -> None:
        """发现态收到广播帧（7.1.2）。"""
        if isinstance(event.data, BroadcastFrame):
            self.callback.on_broadcast_frame(event.data)

    def _handle_send_access_request(self, event: Event) -> None:
        """发现态发起接入请求（7.1.3）。"""
        self._transition(LinkState.ACCESSING)
        if isinstance(event.data, dict):
            self.peer_address = event.data.get("peer_address", b"")
            requested_role = event.data.get("role", Role.T_NODE)
            self.role = requested_role

    def _handle_access_response_received(self, event: Event) -> None:
        """接入态收到接入响应（7.1.3d）。"""
        if isinstance(event.data, dict) and event.data.get("accepted", False):
            role = event.data.get("role")
            if isinstance(role, Role):
                self.role = role
            self._transition(LinkState.CONNECTED)
            self._last_rx_time = time.monotonic()
            self.callback.on_access_accepted(self.role)
        else:
            self._transition(LinkState.SCANNING)
            self.callback.on_access_rejected()

    def _handle_access_request_received(self, event: Event) -> None:
        """广播态收到接入请求（7.1.3c），广播设备侧。"""
        if isinstance(event.data, dict) and event.data.get("accepted", False):
            role = event.data.get("role", Role.G_NODE)
            if isinstance(role, Role):
                self.role = role
            self._transition(LinkState.CONNECTED)
            self._last_rx_time = time.monotonic()
            self.callback.on_access_accepted(self.role)

    def _handle_signaling_received(self, event: Event) -> None:
        """链接态收到控制面信令（7.2）。"""
        self._last_rx_time = time.monotonic()
        if isinstance(event.data, ControlFrame):
            msg = decode_signaling(event.data)
            self.callback.on_signaling(msg)
        else:
            self.callback.on_signaling(event.data)

    def _handle_signaling_send(self, event: Event) -> None:
        """链接态发送控制面信令。

        实际发送由上层完成，此处仅编码并返回 ControlFrame。
        """
        if event.data is not None:
            frame = encode_signaling(event.data)
            return frame
        return None

    def _handle_disconnect_request(self, event: Event) -> None:
        """本端请求断开（7.2.17）。"""
        self._transition(LinkState.DISCONNECTED)
        self.callback.on_disconnected(DisconnectReason.LOCAL_REQUEST)

    def _handle_disconnect_received(self, event: Event) -> None:
        """收到对端断开指示（7.2.17）。"""
        self._transition(LinkState.DISCONNECTED)
        self.callback.on_disconnected(DisconnectReason.REMOTE_REQUEST)

    def _handle_supervision_timeout(self, event: Event) -> None:
        """监督超时（链接态）。"""
        self._transition(LinkState.DISCONNECTED)
        self.callback.on_disconnected(DisconnectReason.TIMEOUT)

    def _handle_start_pairing(self, event: Event) -> None:
        """链接态发起配对流程（9.2）。"""
        self._transition(LinkState.PAIRING)

    def _handle_pairing_complete(self, event: Event) -> None:
        """配对完成，返回链接态（加密）。"""
        self._transition(LinkState.CONNECTED)

    def _handle_pairing_failed(self, event: Event) -> None:
        """配对失败，断开链路。"""
        self._transition(LinkState.DISCONNECTED)
        self.callback.on_disconnected(DisconnectReason.LOCAL_REQUEST)

    def _handle_sleep_request(self, event: Event) -> None:
        """链接态发起休眠（7.2.14）。"""
        self._transition(LinkState.DORMANT)
        self.callback.on_dormant()

    def _handle_wake_request(self, event: Event) -> None:
        """本端请求唤醒（7.2.14）。"""
        self._transition(LinkState.CONNECTED)
        self._last_rx_time = time.monotonic()
        self.callback.on_wakeup()

    def _handle_wake_received(self, event: Event) -> None:
        """收到对端唤醒指示（7.2.14）。"""
        self._transition(LinkState.CONNECTED)
        self._last_rx_time = time.monotonic()
        self.callback.on_wakeup()

    def _handle_dormant_disconnect(self, event: Event) -> None:
        """休眠态断开链路。"""
        self._transition(LinkState.DISCONNECTED)
        self.callback.on_disconnected(DisconnectReason.LOCAL_REQUEST)

    # ---------------------------------------------------------------
    # 链接态便利方法 (7.2.x)
    # ---------------------------------------------------------------

    def send_signaling(self, msg: Any) -> ControlFrame | None:
        """在链接态发送一条控制面信令，返回编码后的 ControlFrame。"""
        if self.state != LinkState.CONNECTED:
            raise InvalidState(
                f"仅链接态可发送信令 (当前 {self.state.name})"
            )
        return encode_signaling(msg)

    def update_params(self, **kwargs: Any) -> None:
        """更新链路参数。"""
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)

    def switch_role(self) -> None:
        """角色切换（7.2.15），仅交换本地角色标记。"""
        if self.state != LinkState.CONNECTED:
            raise InvalidState("角色切换仅在链接态可用")
        if self.role == Role.G_NODE:
            self.role = Role.T_NODE
        elif self.role == Role.T_NODE:
            self.role = Role.G_NODE

    def check_supervision_timeout(self) -> bool:
        """检查监督定时器是否超时。返回 True 表示已超时并触发断开。"""
        if self.state != LinkState.CONNECTED:
            return False
        if self._last_rx_time == 0:
            return False
        elapsed_ms = (time.monotonic() - self._last_rx_time) * 1000
        if elapsed_ms > self.params.supervision_timeout:
            self.process_event(Event(EventType.SUPERVISION_TIMEOUT))
            return True
        return False

    # ---------------------------------------------------------------
    # 7.2.4 - 7.2.12 控制面流程便捷方法
    # ---------------------------------------------------------------

    def request_feature_exchange(self, feature_set: int) -> ControlFrame | None:
        """发起特性交互 (7.2.4)

        :param feature_set: 本端支持特性集 (80-bit 位图)
        """
        from nearlink_sdr.mac.link_control import FeatureExchangeRequest
        msg = FeatureExchangeRequest(feature_set=feature_set)
        return self.send_signaling(msg)

    def respond_feature_exchange(self, feature_set: int) -> ControlFrame | None:
        """回复特性交互 (7.2.4)"""
        from nearlink_sdr.mac.link_control import FeatureExchangeResponse
        msg = FeatureExchangeResponse(feature_set=feature_set)
        return self.send_signaling(msg)

    def request_version_exchange(
        self, spec_version: int = 0, company_id: int = 0, sub_version: int = 0,
    ) -> ControlFrame | None:
        """发起版本交互 (7.2.5)"""
        from nearlink_sdr.mac.link_control import VersionExchange
        msg = VersionExchange(
            spec_version=spec_version,
            company_id=company_id,
            sub_version=sub_version,
        )
        return self.send_signaling(msg)

    def request_data_length_update(
        self, max_rx_bytes: int = 251, max_rx_time: int = 2120,
        max_tx_bytes: int = 251, max_tx_time: int = 2120,
    ) -> ControlFrame | None:
        """发起数据长度更新 (7.2.6)"""
        from nearlink_sdr.mac.link_control import DataLengthRequest
        msg = DataLengthRequest(
            max_rx_bytes=max_rx_bytes, max_rx_time=max_rx_time,
            max_tx_bytes=max_tx_bytes, max_tx_time=max_tx_time,
        )
        return self.send_signaling(msg)

    def respond_data_length_update(
        self, max_rx_bytes: int = 251, max_rx_time: int = 2120,
        max_tx_bytes: int = 251, max_tx_time: int = 2120,
    ) -> ControlFrame | None:
        """回复数据长度更新 (7.2.6)"""
        from nearlink_sdr.mac.link_control import DataLengthResponse
        msg = DataLengthResponse(
            max_rx_bytes=max_rx_bytes, max_rx_time=max_rx_time,
            max_tx_bytes=max_tx_bytes, max_tx_time=max_tx_time,
        )
        return self.send_signaling(msg)

    def configure_channel_report(
        self, enable: int = 1, min_interval: int = 5, max_delay: int = 5,
    ) -> ControlFrame | None:
        """配置信道质量上报 (7.2.7), 由 G 节点发送"""
        from nearlink_sdr.mac.link_control import ChannelReportConfig
        msg = ChannelReportConfig(
            enable=enable, min_interval=min_interval, max_delay=max_delay,
        )
        return self.send_signaling(msg)

    def update_hop_table(
        self, effective_slot: int = 0,
        channel_count: int = 0, channel_table: bytes = b"",
    ) -> ControlFrame | None:
        """更新跳频频点表 (7.2.8), 由 G 节点发送"""
        from nearlink_sdr.mac.link_control import HopTableUpdate
        msg = HopTableUpdate(
            effective_slot=effective_slot,
            channel_count=channel_count,
            channel_table=channel_table,
        )
        return self.send_signaling(msg)

    def update_hop_map(
        self, effective_slot: int = 0, hop_map: bytes = b"",
    ) -> ControlFrame | None:
        """更新跳频地图 (7.2.9), 由 G 节点发送"""
        from nearlink_sdr.mac.link_control import HopMapUpdate
        msg = HopMapUpdate(
            effective_slot=effective_slot,
            hop_map=hop_map,
        )
        return self.send_signaling(msg)

    def request_min_channels(
        self, frame_type: int = 0, bandwidth: int = 0,
        pilot_density: int = 0, min_channels: int = 2,
    ) -> ControlFrame | None:
        """最少可用信道请求 (7.2.10), 由 T 节点发送"""
        from nearlink_sdr.mac.link_control import MinAvailableChannels
        msg = MinAvailableChannels(
            frame_type=frame_type,
            bandwidth=bandwidth,
            pilot_density=pilot_density,
            min_channels=min_channels,
        )
        return self.send_signaling(msg)

    def request_crc_switch(
        self, link_id: int = 0, tx_crc_type: int = 0, rx_crc_type: int = 0,
        tx_crc_init: int = 0, rx_crc_init: int = 0,
    ) -> ControlFrame | None:
        """请求 CRC 切换 (7.2.11)"""
        from nearlink_sdr.mac.link_control import CrcSwitchRequest
        msg = CrcSwitchRequest(
            link_id=link_id,
            tx_crc_type=tx_crc_type,
            rx_crc_type=rx_crc_type,
            tx_crc_init=tx_crc_init,
            rx_crc_init=rx_crc_init,
        )
        return self.send_signaling(msg)

    def indicate_crc_switch(
        self, link_id: int = 0, tx_crc_type: int = 0, rx_crc_type: int = 0,
        tx_crc_init: int = 0, rx_crc_init: int = 0, effective_slot: int = 0,
    ) -> ControlFrame | None:
        """指示 CRC 切换 (7.2.11), 由 G 节点直接切换"""
        from nearlink_sdr.mac.link_control import CrcSwitchIndication
        msg = CrcSwitchIndication(
            link_id=link_id,
            tx_crc_type=tx_crc_type,
            rx_crc_type=rx_crc_type,
            tx_crc_init=tx_crc_init,
            rx_crc_init=rx_crc_init,
            effective_slot=effective_slot,
        )
        return self.send_signaling(msg)

    def request_phy_update(
        self, tx_frame_type: int = 0, rx_frame_type: int = 0,
        tx_bandwidth: int = 0, rx_bandwidth: int = 0,
        tx_pilot_density: int = 0, rx_pilot_density: int = 0,
        tx_feedback_type: int = 0, rx_feedback_type: int = 0,
    ) -> ControlFrame | None:
        """请求物理层类型更新 (7.2.12)"""
        from nearlink_sdr.mac.link_control import PhyUpdateRequest
        msg = PhyUpdateRequest(
            tx_frame_type=tx_frame_type, rx_frame_type=rx_frame_type,
            tx_bandwidth=tx_bandwidth, rx_bandwidth=rx_bandwidth,
            tx_pilot_density=tx_pilot_density, rx_pilot_density=rx_pilot_density,
            tx_feedback_type=tx_feedback_type, rx_feedback_type=rx_feedback_type,
        )
        return self.send_signaling(msg)

    def indicate_phy_update(
        self, tx_frame_type: int = 0, rx_frame_type: int = 0,
        tx_bandwidth: int = 0, rx_bandwidth: int = 0,
        tx_pilot_density: int = 0, rx_pilot_density: int = 0,
        tx_feedback_type: int = 0, rx_feedback_type: int = 0,
        effective_slot: int = 0,
    ) -> ControlFrame | None:
        """指示物理层类型更新 (7.2.12), 由 G 节点直接更新"""
        from nearlink_sdr.mac.link_control import PhyUpdateIndication
        msg = PhyUpdateIndication(
            tx_frame_type=tx_frame_type, rx_frame_type=rx_frame_type,
            tx_bandwidth=tx_bandwidth, rx_bandwidth=rx_bandwidth,
            tx_pilot_density=tx_pilot_density, rx_pilot_density=rx_pilot_density,
            tx_feedback_type=tx_feedback_type, rx_feedback_type=rx_feedback_type,
            effective_slot=effective_slot,
        )
        return self.send_signaling(msg)

    # ---------------------------------------------------------------
    # 7.2.15 角色切换 (带信令)
    # ---------------------------------------------------------------

    def request_role_switch(
        self, effective_slot: int = 0,
    ) -> ControlFrame | None:
        """发起角色切换请求 (7.2.15), 发送 RoleSwitchRequest 信令"""
        from nearlink_sdr.mac.link_control import RoleSwitchRequest
        msg = RoleSwitchRequest(effective_slot=effective_slot)
        return self.send_signaling(msg)

    def execute_role_switch(self, effective_slot: int = 0) -> ControlFrame | None:
        """执行角色切换: 发送信令并交换本地角色标记"""
        frame = self.request_role_switch(effective_slot)
        self.switch_role()
        return frame

    # ---------------------------------------------------------------
    # 7.2.16 PING 流程
    # ---------------------------------------------------------------

    def send_ping(self) -> ControlFrame | None:
        """发起 PING 请求 (7.2.16)"""
        from nearlink_sdr.mac.link_control import PingRequest
        return self.send_signaling(PingRequest())

    def respond_ping(self) -> ControlFrame | None:
        """回复 PING 响应 (7.2.16)"""
        from nearlink_sdr.mac.link_control import PingResponse
        return self.send_signaling(PingResponse())

    # ---------------------------------------------------------------
    # 7.2.17 链路断开 (信令通知)
    # ---------------------------------------------------------------

    def request_disconnect(self) -> None:
        """发起链路断开 (7.2.17)

        驱动状态机进入 DISCONNECTED 状态。
        """
        self.process_event(Event(EventType.DISCONNECT_REQUEST))

    # ---------------------------------------------------------------
    # 7.2.18 链接态异步链路参数更新
    # ---------------------------------------------------------------

    def request_async_param_update(
        self,
        event_group_period_min: int = 0,
        event_group_period_max: int = 0,
        delay_period: int = 0,
        timeout: int = 0,
        expected_period_unit: int = 0,
        effective_ref_slot: int = 0,
        offsets: tuple[int, ...] = (),
        time_slot_length: int = 0,
        time_slot_count: int = 0,
    ) -> ControlFrame | None:
        """发起异步链路参数更新请求 (7.2.18)"""
        from nearlink_sdr.mac.link_control import AsyncLinkParamRequest
        msg = AsyncLinkParamRequest(
            event_group_period_min=event_group_period_min,
            event_group_period_max=event_group_period_max,
            delay_period=delay_period,
            timeout=timeout,
            expected_period_unit=expected_period_unit,
            effective_ref_slot=effective_ref_slot,
            offsets=offsets,
            time_slot_length=time_slot_length,
            time_slot_count=time_slot_count,
        )
        return self.send_signaling(msg)

    def respond_async_param_update(
        self,
        event_group_period_min: int = 0,
        event_group_period_max: int = 0,
        delay_period: int = 0,
        timeout: int = 0,
        expected_period_unit: int = 0,
        effective_ref_slot: int = 0,
        offsets: tuple[int, ...] = (),
        time_slot_length: int = 0,
        time_slot_count: int = 0,
    ) -> ControlFrame | None:
        """回复异步链路参数更新 (7.2.18)"""
        from nearlink_sdr.mac.link_control import AsyncLinkParamResponse
        msg = AsyncLinkParamResponse(
            event_group_period_min=event_group_period_min,
            event_group_period_max=event_group_period_max,
            delay_period=delay_period,
            timeout=timeout,
            expected_period_unit=expected_period_unit,
            effective_ref_slot=effective_ref_slot,
            offsets=offsets,
            time_slot_length=time_slot_length,
            time_slot_count=time_slot_count,
        )
        return self.send_signaling(msg)

    # ---------------------------------------------------------------
    # 7.2.19 同步等时链路管理
    # ---------------------------------------------------------------

    def request_isochronous_link_setup(
        self,
        event_group_set_id: int = 0,
        event_group_id: int = 0,
        effective_slot: int = 0,
        event_group_period: int = 0,
        event_period: int = 0,
        intra_event_interval: int = 0,
        inter_event_interval: int = 0,
        event_count: int = 1,
        sync_anchor_delay: int = 0,
        sync_ref_delay: int = 0,
        scheduling_slot: int = 0,
        tx_rx_indication: int = 0,
        tx_adapt_mode: int = 0,
        rx_adapt_mode: int = 0,
        tx_link_id: int = 0,
        rx_link_id: int = 0,
        tx_frame_type: int = 0,
        rx_frame_type: int = 0,
        tx_bandwidth: int = 0,
        rx_bandwidth: int = 0,
        tx_pilot_density: int = 0,
        rx_pilot_density: int = 0,
        tx_sdu_max: int = 0,
        rx_sdu_max: int = 0,
        tx_sdu_period: int = 0,
        rx_sdu_period: int = 0,
        tx_pdu_max: int = 0,
        rx_pdu_max: int = 0,
        tx_max_time_offset: int = 0,
        rx_max_time_offset: int = 0,
        tx_new_pkt_count: int = 1,
        rx_new_pkt_count: int = 1,
        tx_crc_init: int = 0,
        rx_crc_init: int = 0,
        tx_discard_period: int = 3,
        rx_discard_period: int = 3,
        tx_crc_type: int = 0,
        rx_crc_type: int = 0,
        tx_feedback_type: int = 0,
        rx_feedback_type: int = 0,
    ) -> ControlFrame | None:
        """发起同步等时链路建链 (7.2.19.1)"""
        from nearlink_sdr.mac.link_control import IsochronousLinkSetup
        msg = IsochronousLinkSetup(
            event_group_set_id=event_group_set_id,
            event_group_id=event_group_id,
            effective_slot=effective_slot,
            event_group_period=event_group_period,
            event_period=event_period,
            intra_event_interval=intra_event_interval,
            inter_event_interval=inter_event_interval,
            event_count=event_count,
            sync_anchor_delay=sync_anchor_delay,
            sync_ref_delay=sync_ref_delay,
            scheduling_slot=scheduling_slot,
            tx_rx_indication=tx_rx_indication,
            tx_adapt_mode=tx_adapt_mode,
            rx_adapt_mode=rx_adapt_mode,
            tx_link_id=tx_link_id,
            rx_link_id=rx_link_id,
            tx_frame_type=tx_frame_type,
            rx_frame_type=rx_frame_type,
            tx_bandwidth=tx_bandwidth,
            rx_bandwidth=rx_bandwidth,
            tx_pilot_density=tx_pilot_density,
            rx_pilot_density=rx_pilot_density,
            tx_sdu_max=tx_sdu_max,
            rx_sdu_max=rx_sdu_max,
            tx_sdu_period=tx_sdu_period,
            rx_sdu_period=rx_sdu_period,
            tx_pdu_max=tx_pdu_max,
            rx_pdu_max=rx_pdu_max,
            tx_max_time_offset=tx_max_time_offset,
            rx_max_time_offset=rx_max_time_offset,
            tx_new_pkt_count=tx_new_pkt_count,
            rx_new_pkt_count=rx_new_pkt_count,
            tx_crc_init=tx_crc_init,
            rx_crc_init=rx_crc_init,
            tx_discard_period=tx_discard_period,
            rx_discard_period=rx_discard_period,
            tx_crc_type=tx_crc_type,
            rx_crc_type=rx_crc_type,
            tx_feedback_type=tx_feedback_type,
            rx_feedback_type=rx_feedback_type,
        )
        return self.send_signaling(msg)

    def request_isochronous_param_exchange(
        self,
        event_group_set_id: int = 0,
        event_group_id: int = 0,
        event_group_period: int = 0,
        event_period: int = 0,
        intra_event_interval: int = 0,
        inter_event_interval: int = 0,
        event_count: int = 1,
        sync_anchor_delay: int = 0,
        sync_ref_delay: int = 0,
        param_tag_id: int = 0,
        tx_rx_indication: int = 0,
        tx_adapt_mode: int = 0,
        rx_adapt_mode: int = 0,
        tx_link_id: int = 0,
        rx_link_id: int = 0,
        tx_frame_type: int = 0,
        rx_frame_type: int = 0,
        tx_bandwidth: int = 0,
        rx_bandwidth: int = 0,
        tx_pilot_density: int = 0,
        rx_pilot_density: int = 0,
        tx_sdu_max: int = 0,
        rx_sdu_max: int = 0,
        tx_sdu_period: int = 0,
        rx_sdu_period: int = 0,
        tx_pdu_max: int = 0,
        rx_pdu_max: int = 0,
        tx_max_time_offset: int = 0,
        rx_max_time_offset: int = 0,
        tx_new_pkt_count: int = 1,
        rx_new_pkt_count: int = 1,
        tx_crc_init: int = 0,
        rx_crc_init: int = 0,
        tx_discard_period: int = 3,
        rx_discard_period: int = 3,
        tx_crc_type: int = 0,
        rx_crc_type: int = 0,
        tx_feedback_type: int = 0,
        rx_feedback_type: int = 0,
    ) -> ControlFrame | None:
        """发起同步等时参数交互请求 (7.2.19.2)"""
        from nearlink_sdr.mac.link_control import IsochronousParamExchangeRequest
        msg = IsochronousParamExchangeRequest(
            event_group_set_id=event_group_set_id,
            event_group_id=event_group_id,
            event_group_period=event_group_period,
            event_period=event_period,
            intra_event_interval=intra_event_interval,
            inter_event_interval=inter_event_interval,
            event_count=event_count,
            sync_anchor_delay=sync_anchor_delay,
            sync_ref_delay=sync_ref_delay,
            param_tag_id=param_tag_id,
            tx_rx_indication=tx_rx_indication,
            tx_adapt_mode=tx_adapt_mode,
            rx_adapt_mode=rx_adapt_mode,
            tx_link_id=tx_link_id,
            rx_link_id=rx_link_id,
            tx_frame_type=tx_frame_type,
            rx_frame_type=rx_frame_type,
            tx_bandwidth=tx_bandwidth,
            rx_bandwidth=rx_bandwidth,
            tx_pilot_density=tx_pilot_density,
            rx_pilot_density=rx_pilot_density,
            tx_sdu_max=tx_sdu_max,
            rx_sdu_max=rx_sdu_max,
            tx_sdu_period=tx_sdu_period,
            rx_sdu_period=rx_sdu_period,
            tx_pdu_max=tx_pdu_max,
            rx_pdu_max=rx_pdu_max,
            tx_max_time_offset=tx_max_time_offset,
            rx_max_time_offset=rx_max_time_offset,
            tx_new_pkt_count=tx_new_pkt_count,
            rx_new_pkt_count=rx_new_pkt_count,
            tx_crc_init=tx_crc_init,
            rx_crc_init=rx_crc_init,
            tx_discard_period=tx_discard_period,
            rx_discard_period=rx_discard_period,
            tx_crc_type=tx_crc_type,
            rx_crc_type=rx_crc_type,
            tx_feedback_type=tx_feedback_type,
            rx_feedback_type=rx_feedback_type,
        )
        return self.send_signaling(msg)

    def respond_isochronous_param_exchange(
        self, **kwargs: int,
    ) -> ControlFrame | None:
        """回复同步等时参数交互 (7.2.19.2)"""
        from nearlink_sdr.mac.link_control import IsochronousParamExchangeResponse
        defaults = {
            "event_group_set_id": 0, "event_group_id": 0,
            "event_group_period": 0, "event_period": 0,
            "intra_event_interval": 0, "inter_event_interval": 0,
            "event_count": 1, "sync_anchor_delay": 0, "sync_ref_delay": 0,
            "param_tag_id": 0, "tx_rx_indication": 0,
            "tx_adapt_mode": 0, "rx_adapt_mode": 0,
            "tx_link_id": 0, "rx_link_id": 0,
            "tx_frame_type": 0, "rx_frame_type": 0,
            "tx_bandwidth": 0, "rx_bandwidth": 0,
            "tx_pilot_density": 0, "rx_pilot_density": 0,
            "tx_sdu_max": 0, "rx_sdu_max": 0,
            "tx_sdu_period": 0, "rx_sdu_period": 0,
            "tx_pdu_max": 0, "rx_pdu_max": 0,
            "tx_max_time_offset": 0, "rx_max_time_offset": 0,
            "tx_new_pkt_count": 1, "rx_new_pkt_count": 1,
            "tx_crc_init": 0, "rx_crc_init": 0,
            "tx_discard_period": 3, "rx_discard_period": 3,
            "tx_crc_type": 0, "rx_crc_type": 0,
            "tx_feedback_type": 0, "rx_feedback_type": 0,
        }
        defaults.update(kwargs)
        msg = IsochronousParamExchangeResponse(**defaults)
        return self.send_signaling(msg)

    def request_isochronous_param_update(
        self,
        param_tag_id: int = 0,
        event_group_set_id: int = 0,
        event_group_id: int = 0,
    ) -> ControlFrame | None:
        """发起同步等时参数更新请求 (7.2.19.3)"""
        from nearlink_sdr.mac.link_control import IsochronousParamUpdateRequest
        msg = IsochronousParamUpdateRequest(
            param_tag_id=param_tag_id,
            event_group_set_id=event_group_set_id,
            event_group_id=event_group_id,
        )
        return self.send_signaling(msg)

    def indicate_isochronous_param_update(
        self,
        param_tag_id: int = 0,
        event_group_set_id: int = 0,
        event_group_id: int = 0,
        effective_ref_slot: int = 0,
        event_group_offset: int = 0,
    ) -> ControlFrame | None:
        """发送同步等时参数更新指示 (7.2.19.3)"""
        from nearlink_sdr.mac.link_control import IsochronousParamUpdateIndication
        msg = IsochronousParamUpdateIndication(
            param_tag_id=param_tag_id,
            event_group_set_id=event_group_set_id,
            event_group_id=event_group_id,
            effective_ref_slot=effective_ref_slot,
            event_group_offset=event_group_offset,
        )
        return self.send_signaling(msg)

    # -------------------------------------------------------------------
    # 7.2.20  广播链路管理
    # -------------------------------------------------------------------

    def indicate_broadcast_link_setup(self, **kwargs) -> ControlFrame | None:
        """发送链接态广播链路建立指示 (7.2.20)"""
        from nearlink_sdr.mac.link_control import BroadcastLinkSetup
        defaults: dict = {
            "transmission_type": 0, "adapt_mode": 0,
            "event_group_set_id": 0, "event_group_count": 1,
            "event_group_id": 0, "effective_slot": 0,
            "event_group_interval": 1, "event_group_period": 100,
            "event_period": 10, "event_count": 1,
            "base_link_id": 0, "frame_type": 0, "bandwidth": 0,
            "pilot_density": 0, "sdu_max": 251, "sdu_period": 0,
            "pdu_max": 251, "new_pkt_count": 1, "crc_type": 0,
            "crc_base_init": 0, "hop_map": b"\xff" * 10,
            "sync_anchor_delay": 0, "sync_ref_delay": 0,
        }
        defaults.update(kwargs)
        return self.send_signaling(BroadcastLinkSetup(**defaults))

    def indicate_broadcast_link_param_update(
        self, **kwargs,
    ) -> ControlFrame | None:
        """发送广播链路参数更新指示 (7.2.20)"""
        from nearlink_sdr.mac.link_control import BroadcastLinkParamUpdate
        defaults: dict = {
            "event_group_set_id": 0, "event_group_count": 1,
            "event_group_id": 0, "event_group_period": 100,
            "event_period": 10, "event_count": 1,
            "frame_type": 0, "bandwidth": 0, "pilot_density": 0,
            "sdu_max": 251, "new_pkt_count": 1, "adapt_mode": 0,
            "sdu_period": 0, "pdu_max": 251, "crc_type": 0,
            "crc_base_init": 0, "sync_anchor_delay": 0,
            "sync_ref_delay": 0, "effective_ref_slot": 0,
            "event_group_offset": 0,
        }
        defaults.update(kwargs)
        return self.send_signaling(BroadcastLinkParamUpdate(**defaults))

    def update_broadcast_hop_map(
        self,
        hop_map: bytes = b"\xff" * 10,
        effective_slot: int = 0,
    ) -> ControlFrame | None:
        """发送广播链路跳频地图更新指示 (7.2.20)"""
        from nearlink_sdr.mac.link_control import BroadcastHopMapUpdate
        return self.send_signaling(
            BroadcastHopMapUpdate(hop_map=hop_map,
                                  effective_slot=effective_slot))

    def indicate_broadcast_link_disconnect(
        self,
        link_id: int = 0,
        error_reason: int = 0,
    ) -> ControlFrame | None:
        """发送广播链路断开指示 (7.2.20)"""
        from nearlink_sdr.mac.link_control import BroadcastLinkDisconnect
        return self.send_signaling(
            BroadcastLinkDisconnect(link_id=link_id,
                                    error_reason=error_reason))

    # -------------------------------------------------------------------
    # 7.2.21  系统管理帧链路管理
    # -------------------------------------------------------------------

    def request_smf_param_update(
        self,
        smf_period: int = 100,
        smf_start_offset: int = 0,
        link_id: int = 0,
        frame_type: int = 0,
        bandwidth: int = 0,
        pilot_density: int = 0,
        crc_type: int = 0,
    ) -> ControlFrame | None:
        """发送系统管理帧参数更新请求 (7.2.21)"""
        from nearlink_sdr.mac.link_control import SMFParamUpdateRequest
        return self.send_signaling(SMFParamUpdateRequest(
            smf_period=smf_period, smf_start_offset=smf_start_offset,
            link_id=link_id, frame_type=frame_type, bandwidth=bandwidth,
            pilot_density=pilot_density, crc_type=crc_type))

    def indicate_smf_param_update(
        self,
        smf_period: int = 100,
        smf_start_offset: int = 0,
        link_id: int = 0,
        frame_type: int = 0,
        bandwidth: int = 0,
        pilot_density: int = 0,
        crc_type: int = 0,
        crc_init: int = 0,
    ) -> ControlFrame | None:
        """发送系统管理帧参数更新指示 (7.2.21)"""
        from nearlink_sdr.mac.link_control import SMFParamUpdateIndication
        return self.send_signaling(SMFParamUpdateIndication(
            smf_period=smf_period, smf_start_offset=smf_start_offset,
            link_id=link_id, frame_type=frame_type, bandwidth=bandwidth,
            pilot_density=pilot_density, crc_type=crc_type,
            crc_init=crc_init))

    def request_smf_timeslot_update(
        self,
        link_id: int = 0,
        current_offset: int = 0,
        offsets: tuple[int, ...] = (0, 0, 0, 0),
    ) -> ControlFrame | None:
        """发送系统管理帧时间片更新请求 (7.2.21)"""
        from nearlink_sdr.mac.link_control import SMFTimeSlotUpdateRequest
        return self.send_signaling(SMFTimeSlotUpdateRequest(
            link_id=link_id, current_offset=current_offset,
            offsets=offsets))

    def respond_smf_timeslot_update(
        self,
        link_id: int = 0,
        offset: int = 0,
        effective_slot: int = 0,
    ) -> ControlFrame | None:
        """发送系统管理帧时间片更新响应 (7.2.21)"""
        from nearlink_sdr.mac.link_control import SMFTimeSlotUpdateResponse
        return self.send_signaling(SMFTimeSlotUpdateResponse(
            link_id=link_id, offset=offset,
            effective_slot=effective_slot))

    def terminate_smf_signaling(
        self,
        terminate_type: int = 0,
    ) -> ControlFrame | None:
        """发送系统管理帧信令传输终止 (7.2.21)"""
        from nearlink_sdr.mac.link_control import SMFSignalingTerminate
        return self.send_signaling(
            SMFSignalingTerminate(terminate_type=terminate_type))

    # -------------------------------------------------------------------
    # 7.2.22  异步组播链路管理
    # -------------------------------------------------------------------

    def indicate_async_multicast_link_setup(
        self, **kwargs,
    ) -> ControlFrame | None:
        """发送异步组播链路建链指示 (7.2.22)"""
        from nearlink_sdr.mac.link_control import AsyncMulticastLinkSetup
        defaults: dict = {
            "event_group_set_id": 0, "event_group_id": 0,
            "effective_slot": 0, "event_group_period": 100,
            "event_period": 10, "intra_event_interval": 0,
            "inter_event_interval": 0, "scheduling_slot": 0,
            "tx_rx_indication": 0, "tx_link_id": 0, "rx_link_id": 0,
            "tx_frame_type": 0, "rx_frame_type": 0,
            "tx_bandwidth": 0, "rx_bandwidth": 0,
            "tx_pilot_density": 0, "rx_pilot_density": 0,
            "tx_sdu_max": 251, "rx_sdu_max": 251,
            "tx_sdu_period": 0, "rx_sdu_period": 0,
            "tx_pdu_max": 251, "rx_pdu_max": 251,
            "tx_max_time_offset": 0, "rx_max_time_offset": 0,
            "tx_crc_init": 0, "rx_crc_init": 0,
            "tx_crc_type": 0, "rx_crc_type": 0,
            "tx_feedback_type": 0, "rx_feedback_type": 0,
        }
        defaults.update(kwargs)
        return self.send_signaling(AsyncMulticastLinkSetup(**defaults))

    def request_async_multicast_param_exchange(
        self,
        payload: bytes = b"\x00" * 41,
    ) -> ControlFrame | None:
        """发送异步组播链路参数交互请求 (7.2.22)"""
        from nearlink_sdr.mac.link_control import (
            AsyncMulticastParamExchangeRequest,
        )
        return self.send_signaling(
            AsyncMulticastParamExchangeRequest(payload=payload))

    def respond_async_multicast_param_exchange(
        self,
        payload: bytes = b"\x00" * 41,
    ) -> ControlFrame | None:
        """发送异步组播链路参数交互响应 (7.2.22)"""
        from nearlink_sdr.mac.link_control import (
            AsyncMulticastParamExchangeResponse,
        )
        return self.send_signaling(
            AsyncMulticastParamExchangeResponse(payload=payload))

    def request_async_multicast_param_update(
        self,
        param_tag_id: int = 0,
        event_group_set_id: int = 0,
        event_group_id: int = 0,
    ) -> ControlFrame | None:
        """发送异步组播链路参数更新请求 (7.2.22)"""
        from nearlink_sdr.mac.link_control import (
            AsyncMulticastParamUpdateRequest,
        )
        return self.send_signaling(AsyncMulticastParamUpdateRequest(
            param_tag_id=param_tag_id,
            event_group_set_id=event_group_set_id,
            event_group_id=event_group_id))

    def indicate_async_multicast_param_update(
        self,
        param_tag_id: int = 0,
        event_group_set_id: int = 0,
        event_group_id: int = 0,
        effective_ref_slot: int = 0,
        event_group_offset: int = 0,
    ) -> ControlFrame | None:
        """发送异步组播链路参数更新指示 (7.2.22)"""
        from nearlink_sdr.mac.link_control import (
            AsyncMulticastParamUpdateIndication,
        )
        return self.send_signaling(AsyncMulticastParamUpdateIndication(
            param_tag_id=param_tag_id,
            event_group_set_id=event_group_set_id,
            event_group_id=event_group_id,
            effective_ref_slot=effective_ref_slot,
            event_group_offset=event_group_offset))

    def indicate_async_multicast_reconfig(
        self, **kwargs,
    ) -> ControlFrame | None:
        """发送异步组播链路参数重配置指示 (7.2.22)"""
        from nearlink_sdr.mac.link_control import AsyncMulticastReconfig
        defaults: dict = {
            "effective_ref_slot": 0, "event_group_offset": 0,
            "event_group_period": 100, "event_period": 10,
            "delay_period": 0, "timeout": 300,
            "intra_event_interval": 0, "inter_event_interval": 0,
            "event_count": 1, "payload_count": 0,
            "scheduling_slot": 0, "tx_rx_indication": 0,
            "tx_max_pdu": 251, "rx_max_pdu": 251,
            "tx_max_time_offset": 0, "rx_max_time_offset": 0,
        }
        defaults.update(kwargs)
        return self.send_signaling(AsyncMulticastReconfig(**defaults))

    def disconnect_multicast(
        self,
        link_id: int = 0,
    ) -> ControlFrame | None:
        """发送组播链路断开指示 (7.2.22)"""
        from nearlink_sdr.mac.link_control import MulticastDisconnect
        return self.send_signaling(
            MulticastDisconnect(link_id=link_id))

    # -------------------------------------------------------------------
    # 7.2.23  窄带跳频测量
    # -------------------------------------------------------------------

    def request_narrowband_meas_cap(self) -> ControlFrame | None:
        """发送窄带跳频测量能力请求 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandMeasCapRequest
        return self.send_signaling(NarrowbandMeasCapRequest())

    def respond_narrowband_meas_cap(
        self,
        payload: bytes = b"\x00" * 32,
    ) -> ControlFrame | None:
        """发送窄带跳频测量能力响应 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandMeasCapResponse
        return self.send_signaling(
            NarrowbandMeasCapResponse(payload=payload))

    def config_narrowband_meas(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送窄带跳频测量信号配置 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandMeasConfig
        return self.send_signaling(
            NarrowbandMeasConfig(payload=payload))

    def report_narrowband_meas(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送窄带跳频测量信息上报 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandMeasReport
        return self.send_signaling(
            NarrowbandMeasReport(payload=payload))

    def action_narrowband_meas(
        self,
        config_index: int = 0,
        start_slot: int = 0,
        action_config: int = 0,
    ) -> ControlFrame | None:
        """发送窄带跳频测量行为指示 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandMeasAction
        return self.send_signaling(NarrowbandMeasAction(
            config_index=config_index, start_slot=start_slot,
            action_config=action_config))

    def request_coordinate(self) -> ControlFrame | None:
        """发送坐标信息请求 (7.2.23)"""
        from nearlink_sdr.mac.link_control import CoordinateRequest
        return self.send_signaling(CoordinateRequest())

    def report_coordinate(
        self,
        rel_x: int = 0, rel_y: int = 0, rel_z: int = 0,
        abs_lon: int = 0, abs_lat: int = 0, abs_alt: int = 0,
    ) -> ControlFrame | None:
        """发送坐标信息上报 (7.2.23)"""
        from nearlink_sdr.mac.link_control import CoordinateReport
        return self.send_signaling(CoordinateReport(
            rel_x=rel_x, rel_y=rel_y, rel_z=rel_z,
            abs_lon=abs_lon, abs_lat=abs_lat, abs_alt=abs_alt))

    def config_coordinate(
        self,
        rel_x: int = 0, rel_y: int = 0, rel_z: int = 0,
        abs_lon: int = 0, abs_lat: int = 0, abs_alt: int = 0,
    ) -> ControlFrame | None:
        """发送坐标信息配置 (7.2.23)"""
        from nearlink_sdr.mac.link_control import CoordinateConfig
        return self.send_signaling(CoordinateConfig(
            rel_x=rel_x, rel_y=rel_y, rel_z=rel_z,
            abs_lon=abs_lon, abs_lat=abs_lat, abs_alt=abs_alt))

    def request_narrowband_delay(self) -> ControlFrame | None:
        """发送窄带跳频测量时延信息请求 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandDelayRequest
        return self.send_signaling(NarrowbandDelayRequest())

    def respond_narrowband_delay(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送窄带跳频测量时延信息响应 (7.2.23)"""
        from nearlink_sdr.mac.link_control import NarrowbandDelayResponse
        return self.send_signaling(
            NarrowbandDelayResponse(payload=payload))

    # -------------------------------------------------------------------
    # 7.2.24  超宽带脉冲测量与感知
    # -------------------------------------------------------------------

    def request_uwb_meas_cap(self) -> ControlFrame | None:
        """发送超宽带脉冲测量能力请求 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBMeasCapRequest
        return self.send_signaling(UWBMeasCapRequest())

    def respond_uwb_meas_cap(
        self,
        payload: bytes = b"\x00" * 50,
    ) -> ControlFrame | None:
        """发送超宽带脉冲测量能力响应 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBMeasCapResponse
        return self.send_signaling(
            UWBMeasCapResponse(payload=payload))

    def config_uwb_meas(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送超宽带脉冲测量配置 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBMeasConfig
        return self.send_signaling(UWBMeasConfig(payload=payload))

    def feedback_uwb_meas_config(
        self,
        config_index: int = 0,
        status: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲测量配置反馈 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBMeasConfigFeedback
        return self.send_signaling(UWBMeasConfigFeedback(
            config_index=config_index, status=status))

    def report_uwb_meas(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送超宽带脉冲测量信息上报 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBMeasReport
        return self.send_signaling(UWBMeasReport(payload=payload))

    def action_uwb_meas(
        self,
        config_index: int = 0,
        start_slot: int = 0,
        action_config: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲测量行为指示 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBMeasAction
        return self.send_signaling(UWBMeasAction(
            config_index=config_index, start_slot=start_slot,
            action_config=action_config))

    def request_uwb_sensing_cap(self) -> ControlFrame | None:
        """发送超宽带脉冲感知能力请求 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingCapRequest
        return self.send_signaling(UWBSensingCapRequest())

    def respond_uwb_sensing_cap(
        self,
        payload: bytes = b"\x00" * 51,
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知能力响应 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingCapResponse
        return self.send_signaling(
            UWBSensingCapResponse(payload=payload))

    def config_uwb_sensing(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知配置 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingConfig
        return self.send_signaling(
            UWBSensingConfig(payload=payload))

    def feedback_uwb_sensing_config(
        self,
        config_index: int = 0,
        status: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知配置反馈 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingConfigFeedback
        return self.send_signaling(UWBSensingConfigFeedback(
            config_index=config_index, status=status))

    def report_uwb_sensing(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知信息上报 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingReport
        return self.send_signaling(
            UWBSensingReport(payload=payload))

    def action_uwb_sensing(
        self,
        config_index: int = 0,
        start_slot: int = 0,
        action_config: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知行为指示 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingAction
        return self.send_signaling(UWBSensingAction(
            config_index=config_index, start_slot=start_slot,
            action_config=action_config))

    def request_uwb_sensing_process(
        self,
        payload: bytes = b"\x00" * 16,
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知流程请求 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingProcessRequest
        return self.send_signaling(
            UWBSensingProcessRequest(payload=payload))

    def feedback_uwb_sensing_process(
        self,
        process_index: int = 0,
        status: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲感知流程反馈 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBSensingProcessFeedback
        return self.send_signaling(UWBSensingProcessFeedback(
            process_index=process_index, status=status))

    def request_uwb_proxy_sensing(
        self,
        proxy_index: int = 0,
        sensing_index: int = 0,
        meas_quantity: int = 0,
        report_period: int = 0,
        bandwidth: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲代理感知请求 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBProxySensingRequest
        return self.send_signaling(UWBProxySensingRequest(
            proxy_index=proxy_index, sensing_index=sensing_index,
            meas_quantity=meas_quantity, report_period=report_period,
            bandwidth=bandwidth))

    def feedback_uwb_proxy_sensing(
        self,
        proxy_index: int = 0,
        sensing_index: int = 0,
        status: int = 0,
        meas_quantity1: int = 0,
        meas_quantity2: int = 0,
        bandwidth1: int = 0,
        bandwidth2: int = 0,
    ) -> ControlFrame | None:
        """发送超宽带脉冲代理感知反馈 (7.2.24)"""
        from nearlink_sdr.mac.link_control import UWBProxySensingFeedback
        return self.send_signaling(UWBProxySensingFeedback(
            proxy_index=proxy_index, sensing_index=sensing_index,
            status=status, meas_quantity1=meas_quantity1,
            meas_quantity2=meas_quantity2, bandwidth1=bandwidth1,
            bandwidth2=bandwidth2))

    # -------------------------------------------------------------------
    # 7.2.25  窄带跳频感知
    # -------------------------------------------------------------------

    def request_narrowband_sensing(
        self,
        payload: bytes = b"\x00" * 16,
    ) -> ControlFrame | None:
        """发送窄带跳频感知流程请求 (7.2.25)"""
        from nearlink_sdr.mac.link_control import NarrowbandSensingRequest
        return self.send_signaling(
            NarrowbandSensingRequest(payload=payload))

    def feedback_narrowband_sensing(
        self,
        process_index: int = 0,
        status: int = 0,
    ) -> ControlFrame | None:
        """发送窄带跳频感知流程反馈 (7.2.25)"""
        from nearlink_sdr.mac.link_control import NarrowbandSensingFeedback
        return self.send_signaling(NarrowbandSensingFeedback(
            process_index=process_index, status=status))

    def request_narrowband_sensing_cap(self) -> ControlFrame | None:
        """发送窄带跳频感知能力请求 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandSensingCapRequest,
        )
        return self.send_signaling(NarrowbandSensingCapRequest())

    def respond_narrowband_sensing_cap(
        self,
        payload: bytes = b"\x00" * 50,
    ) -> ControlFrame | None:
        """发送窄带跳频感知能力响应 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandSensingCapResponse,
        )
        return self.send_signaling(
            NarrowbandSensingCapResponse(payload=payload))

    def config_narrowband_sensing(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送窄带跳频感知配置 (7.2.25)"""
        from nearlink_sdr.mac.link_control import NarrowbandSensingConfig
        return self.send_signaling(
            NarrowbandSensingConfig(payload=payload))

    def feedback_narrowband_sensing_config(
        self,
        config_index: int = 0,
        status: int = 0,
    ) -> ControlFrame | None:
        """发送窄带跳频感知配置反馈 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandSensingConfigFeedback,
        )
        return self.send_signaling(NarrowbandSensingConfigFeedback(
            config_index=config_index, status=status))

    def report_narrowband_sensing(
        self,
        payload: bytes = b"",
    ) -> ControlFrame | None:
        """发送窄带跳频感知信息上报 (7.2.25)"""
        from nearlink_sdr.mac.link_control import NarrowbandSensingReport
        return self.send_signaling(
            NarrowbandSensingReport(payload=payload))

    def action_narrowband_sensing(
        self,
        config_index: int = 0,
        start_slot: int = 0,
        action_config: int = 0,
    ) -> ControlFrame | None:
        """发送窄带跳频感知行为指示 (7.2.25)"""
        from nearlink_sdr.mac.link_control import NarrowbandSensingAction
        return self.send_signaling(NarrowbandSensingAction(
            config_index=config_index, start_slot=start_slot,
            action_config=action_config))

    def request_narrowband_proxy_sensing(
        self,
        proxy_index: int = 0,
        sensing_index: int = 0,
        meas_quantity: int = 0,
        report_period: int = 0,
        bandwidth: int = 0,
    ) -> ControlFrame | None:
        """发送窄带跳频代理感知请求 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandProxySensingRequest,
        )
        return self.send_signaling(NarrowbandProxySensingRequest(
            proxy_index=proxy_index, sensing_index=sensing_index,
            meas_quantity=meas_quantity, report_period=report_period,
            bandwidth=bandwidth))

    def feedback_narrowband_proxy_sensing(
        self,
        proxy_index: int = 0,
        sensing_index: int = 0,
        status: int = 0,
        meas_quantity1: int = 0,
        meas_quantity2: int = 0,
        bandwidth1: int = 0,
        bandwidth2: int = 0,
    ) -> ControlFrame | None:
        """发送窄带跳频代理感知反馈 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandProxySensingFeedback,
        )
        return self.send_signaling(NarrowbandProxySensingFeedback(
            proxy_index=proxy_index, sensing_index=sensing_index,
            status=status, meas_quantity1=meas_quantity1,
            meas_quantity2=meas_quantity2, bandwidth1=bandwidth1,
            bandwidth2=bandwidth2))

    def request_narrowband_meas_config_update(
        self,
        payload: bytes = b"\x00" * 32,
    ) -> ControlFrame | None:
        """发送窄带跳频测量信号配置更新请求 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandMeasConfigUpdateRequest,
        )
        return self.send_signaling(
            NarrowbandMeasConfigUpdateRequest(payload=payload))

    def indicate_narrowband_meas_config_update(
        self,
        payload: bytes = b"\x00" * 32,
    ) -> ControlFrame | None:
        """发送窄带跳频测量信号配置更新指示 (7.2.25)"""
        from nearlink_sdr.mac.link_control import (
            NarrowbandMeasConfigUpdateIndication,
        )
        return self.send_signaling(
            NarrowbandMeasConfigUpdateIndication(payload=payload))

    # -------------------------------------------------------------------
    # 资源预留
    # -------------------------------------------------------------------

    def reserve_resource(
        self,
        config_index: int = 0,
        effective_slot: int = 0,
        event_group_period: int = 100,
        event_period: int = 10,
        event_length: int = 1,
        event_count: int = 1,
        scheduling_slot: int = 0,
    ) -> ControlFrame | None:
        """发送资源预留指示 (7.3.2.91)"""
        from nearlink_sdr.mac.link_control import ResourceReservation
        return self.send_signaling(ResourceReservation(
            config_index=config_index, effective_slot=effective_slot,
            event_group_period=event_group_period,
            event_period=event_period, event_length=event_length,
            event_count=event_count, scheduling_slot=scheduling_slot))

    def terminate_resource_reservation(
        self,
        config_index: int = 0,
        reason: int = 0,
    ) -> ControlFrame | None:
        """发送资源预留终止 (7.3.2.92)"""
        from nearlink_sdr.mac.link_control import (
            ResourceReservationTerminate,
        )
        return self.send_signaling(ResourceReservationTerminate(
            config_index=config_index, reason=reason))


# -----------------------------------------------------------------------
# 异常
# -----------------------------------------------------------------------

class InvalidTransition(Exception):
    """非法状态转换。"""


class InvalidState(Exception):
    """当前状态不允许此操作。"""


# -----------------------------------------------------------------------
# 事件分发表
# -----------------------------------------------------------------------

_HandlerFunc = type(LinkManager._handle_start_broadcast)

_HANDLERS: dict[tuple[LinkState, EventType], _HandlerFunc] = {
    # IDLE
    (LinkState.IDLE, EventType.START_BROADCAST):
        LinkManager._handle_start_broadcast,
    (LinkState.IDLE, EventType.START_SCAN):
        LinkManager._handle_start_scan,
    # BROADCASTING
    (LinkState.BROADCASTING, EventType.STOP_BROADCAST):
        LinkManager._handle_stop_broadcast,
    (LinkState.BROADCASTING, EventType.ACCESS_REQUEST_RECEIVED):
        LinkManager._handle_access_request_received,
    # SCANNING
    (LinkState.SCANNING, EventType.STOP_SCAN):
        LinkManager._handle_stop_scan,
    (LinkState.SCANNING, EventType.BROADCAST_RECEIVED):
        LinkManager._handle_broadcast_received,
    (LinkState.SCANNING, EventType.SEND_ACCESS_REQUEST):
        LinkManager._handle_send_access_request,
    # ACCESSING
    (LinkState.ACCESSING, EventType.ACCESS_RESPONSE_RECEIVED):
        LinkManager._handle_access_response_received,
    # CONNECTED
    (LinkState.CONNECTED, EventType.SIGNALING_RECEIVED):
        LinkManager._handle_signaling_received,
    (LinkState.CONNECTED, EventType.SIGNALING_SEND):
        LinkManager._handle_signaling_send,
    (LinkState.CONNECTED, EventType.DISCONNECT_REQUEST):
        LinkManager._handle_disconnect_request,
    (LinkState.CONNECTED, EventType.DISCONNECT_RECEIVED):
        LinkManager._handle_disconnect_received,
    (LinkState.CONNECTED, EventType.SUPERVISION_TIMEOUT):
        LinkManager._handle_supervision_timeout,
    # CONNECTED → PAIRING
    (LinkState.CONNECTED, EventType.START_PAIRING):
        LinkManager._handle_start_pairing,
    # CONNECTED → DORMANT (7.2.14)
    (LinkState.CONNECTED, EventType.SLEEP_REQUEST):
        LinkManager._handle_sleep_request,
    # PAIRING
    (LinkState.PAIRING, EventType.PAIRING_COMPLETE):
        LinkManager._handle_pairing_complete,
    (LinkState.PAIRING, EventType.PAIRING_FAILED):
        LinkManager._handle_pairing_failed,
    # DORMANT (7.2.14)
    (LinkState.DORMANT, EventType.WAKE_REQUEST):
        LinkManager._handle_wake_request,
    (LinkState.DORMANT, EventType.WAKE_RECEIVED):
        LinkManager._handle_wake_received,
    (LinkState.DORMANT, EventType.DISCONNECT_REQUEST):
        LinkManager._handle_dormant_disconnect,
}
