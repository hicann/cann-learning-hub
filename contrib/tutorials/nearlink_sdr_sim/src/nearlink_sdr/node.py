"""SLE 节点实体 -- 统一收发接口。

将 MAC/PHY 各模块整合为单一 SLE 节点, 提供完整的链路生命周期管理
和数据收发能力。支持仿真模式和 USRP 硬件模式。
"""

from __future__ import annotations

__all__ = [
    "NodeCallback",
    "NodeConfig",
    "NodeRole",
    "NodeState",
    "RxResult",
    "SleNode",
    "TransportMode",
    "TxResult",
]


import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np
from cryptography.exceptions import InvalidTag

from nearlink_sdr.mac.access import (
    AccessWhitelist,
    BroadcasterAccessManager,
    DiscoveryManager,
    InitiatorAccessManager,
    NonConnectedBroadcastConfig,
    NonConnectedBroadcastManager,
    run_access_procedure,
)
from nearlink_sdr.mac.broadcast import BroadcastFilter, BroadcastFrame
from nearlink_sdr.mac.frame import AsyncDataFrame, ControlFrame
from nearlink_sdr.mac.link_manager import (
    DisconnectReason,
    Event,
    EventType,
    LinkManager,
    LinkManagerCallback,
    LinkParams,
    LinkState,
    Role,
)
from nearlink_sdr.mac.power_control import PowerController
from nearlink_sdr.mac.qos import (
    ArqState,
    FlowController,
    HarqController,
    LinkQualityTracker,
    LinkType,
    Priority,
    QosManager,
    TxDecision,
    TxQueue,
)
from nearlink_sdr.mac.scheduler import (
    EventTimingParams,
    ScheduleManager,
    SmfScheduleConfig,
)
from nearlink_sdr.mac.security_manager import (
    FrameCryptoContext,
    PairingManager,
    PairingState,
)
from nearlink_sdr.mac.smf_scheduler import SMFScheduleParams, SMFTransmitScheduler
from nearlink_sdr.phy.channel import ChannelConfig, ChannelModel
from nearlink_sdr.phy.control_info import ControlInfoA2
from nearlink_sdr.phy.data_link import AsyncDataLinkParams, SyncDataLinkParams
from nearlink_sdr.phy.freq_hopping import (
    FreqTable,
    data_link_hop,
    derive_hop_param2,
)
from nearlink_sdr.phy.mac_interface import (
    MacRxResult,
    iq_to_mac,
    mac_to_iq,
)
from nearlink_sdr.phy.measurement import measurement_signal_1
from nearlink_sdr.phy.sdr_backend import SDRConfig, SDRDevice, create_device
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.phy.usrp import SLETransceiver, USRPConfig, USRPDevice

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ── 节点配置 ──


class NodeRole(IntEnum):
    """节点初始角色偏好。"""
    AUTO = 0
    G_NODE = 1
    T_NODE = 2


class TransportMode(IntEnum):
    """传输模式。"""
    SIMULATION = auto()
    USRP = auto()


@dataclass
class NodeConfig:
    """SLE 节点配置。

    :ivar address: 6 字节设备地址。
    :ivar role: 角色偏好 (AUTO/G_NODE/T_NODE)。
    :ivar frame_type: 帧类型 (1-4)。
    :ivar mcs_index: MCS 索引 (0-12)。
    :ivar bandwidth_mhz: 信道带宽 (1/2/4 MHz)。
    :ivar pilot_interval: 导频插入间隔 (0/4/8/16)。
    :ivar max_pdu: 最大 PDU 长度 (字节)。
    :ivar max_retransmit: 最大重传次数。
    :ivar enable_encryption: 是否启用加密。
    :ivar transport: 传输模式。
    :ivar band: 频段标识。
    :ivar hop_param2: 跳频参数 2 (0 表示自动从地址派生)。
    :ivar blocked_channels: 被阻塞的信道号集合。
    :ivar tx_power_dbm: 初始发射功率 (dBm)。
    :ivar max_power_dbm: 最大发射功率。
    :ivar min_power_dbm: 最小发射功率。
    :ivar channel_config: 仿真模式下的信道模型配置 (None 表示理想信道)。
    :ivar sdr_config: SDR 硬件配置 (transport=USRP 时使用, 支持 mock/uhd/pluto 后端)。
    :ivar usrp_config: (已废弃) USRP 配置, 保留向后兼容; 优先使用 sdr_config。
    :ivar smf_enabled: 是否启用系统管理帧调度。
    """
    address: bytes = b"\x00" * 6
    role: NodeRole = NodeRole.AUTO
    frame_type: int = 2
    mcs_index: int = 7
    bandwidth_mhz: int = 1
    pilot_interval: int = 8
    max_pdu: int = 256
    max_retransmit: int = 3
    enable_encryption: bool = False
    transport: TransportMode = TransportMode.SIMULATION
    band: str = "2400"
    hop_param2: int = 0
    blocked_channels: set[int] = field(default_factory=set)
    tx_power_dbm: float = 0.0
    max_power_dbm: float = 20.0
    min_power_dbm: float = -127.0
    channel_config: ChannelConfig | None = None
    sdr_config: SDRConfig | None = None
    usrp_config: USRPConfig | None = None
    smf_enabled: bool = False


# ── 节点状态 ──


class NodeState(IntEnum):
    """节点高层状态, 简化自 LinkState。"""
    IDLE = 0
    ADVERTISING = auto()
    SCANNING = auto()
    CONNECTING = auto()
    PAIRED = auto()
    CONNECTED = auto()
    DISCONNECTED = auto()


# ── 节点回调 ──


class NodeCallback:
    """节点事件回调, 子类可覆盖。"""

    def on_state_changed(self, old: NodeState, new: NodeState) -> None:
        pass

    def on_connected(self, peer_address: bytes, role: Role) -> None:
        pass

    def on_data_received(self, data: bytes) -> None:
        pass

    def on_disconnected(self, reason: DisconnectReason) -> None:
        pass

    def on_broadcast_received(self, frame: BroadcastFrame) -> None:
        pass

    def on_discovery_complete(self, devices: dict) -> None:
        pass

    def on_measurement_result(self, result: dict) -> None:
        pass


# ── 收发结果 ──


@dataclass
class TxResult:
    """发射结果。"""
    iq: np.ndarray | None
    mac_bytes: bytes | None
    decision: TxDecision
    encrypted: bool = False
    channel: int = -1


@dataclass
class RxResult:
    """接收结果。"""
    data: bytes | None
    success: bool
    decrypted: bool = False


# ── SLE 节点实体 ──


@dataclass
class SleNode:
    """SparkLink SLE 节点实体。

    整合链路管理、安全、QoS、PHY 流水线、跳频、功率控制、
    时序调度、接入流程、信道模型为统一收发接口。

    用法::

        node = SleNode(NodeConfig(address=b"\\x01\\x02\\x03\\x04\\x05\\x06"))
        node.send(b"hello", Priority.NORMAL)
        tx_result = node.transmit()
        # ... 通过信道传输 tx_result.iq ...
        rx_result = node.receive(rx_iq, n_bytes)
    """
    config: NodeConfig = field(default_factory=NodeConfig)
    callback: NodeCallback = field(default_factory=NodeCallback)

    # 内部组件 (延迟初始化)
    _link_mgr: LinkManager = field(init=False, repr=False)
    _qos: QosManager = field(init=False, repr=False)
    _tx_config: TxConfig = field(init=False, repr=False)
    _pairing: PairingManager | None = field(init=False, default=None, repr=False)
    _crypto: FrameCryptoContext | None = field(init=False, default=None, repr=False)
    _state: NodeState = field(init=False, default=NodeState.IDLE)
    _peer_address: bytes = field(init=False, default=b"")
    _tx_count: int = field(init=False, default=0)
    _rx_count: int = field(init=False, default=0)

    # 跳频管理
    _freq_table: FreqTable = field(init=False, repr=False)
    _hop_param2: int = field(init=False, default=0)

    # 功率控制
    _power_ctrl: PowerController = field(init=False, repr=False)

    # 时序调度
    _scheduler: ScheduleManager = field(init=False, repr=False)

    # SMF 调度
    _smf_scheduler: SMFTransmitScheduler | None = field(
        init=False, default=None, repr=False,
    )

    # 接入流程
    _whitelist: AccessWhitelist = field(init=False, repr=False)
    _discovery: DiscoveryManager = field(init=False, repr=False)
    _broadcast_filter: BroadcastFilter = field(init=False, repr=False)
    _broadcaster_mgr: BroadcasterAccessManager | None = field(
        init=False, default=None, repr=False,
    )
    _initiator_mgr: InitiatorAccessManager | None = field(
        init=False, default=None, repr=False,
    )

    # 数据链路参数
    _data_link_params: AsyncDataLinkParams | SyncDataLinkParams | None = field(
        init=False, default=None, repr=False,
    )

    # 信道模型 (仿真模式)
    _channel: ChannelModel | None = field(init=False, default=None, repr=False)

    # SDR 硬件设备
    _sdr_device: SDRDevice | None = field(init=False, default=None, repr=False)

    # 旧版 USRP 收发器 (向后兼容)
    _transceiver: SLETransceiver | None = field(
        init=False, default=None, repr=False,
    )

    def __post_init__(self) -> None:
        cfg = self.config
        self._tx_config = TxConfig(
            frame_type=cfg.frame_type,
            mcs_index=cfg.mcs_index,
            pid=int.from_bytes(cfg.address[:3], "big") if len(cfg.address) >= 3 else 0,
            whitening_seed=0x52,
            crc_seed=0x555555,
            crc_len=24,
            ctrl_bits_len=self._ctrl_bits_len(),
            pilot_interval=cfg.pilot_interval,
        )
        link_type = LinkType.ASYNC if cfg.frame_type == 2 else LinkType.SYNC
        self._qos = QosManager(
            arq=ArqState(link_type=link_type, max_retransmit=cfg.max_retransmit),
            harq=HarqController(),
            flow=FlowController(),
            quality=LinkQualityTracker(),
            tx_queue=TxQueue(max_size=cfg.max_pdu),
        )
        self._link_mgr = LinkManager(
            local_address=cfg.address,
            params=LinkParams(frame_type=cfg.frame_type, bandwidth=cfg.bandwidth_mhz),
            callback=_InternalCallback(self),
        )

        # 跳频
        self._freq_table = FreqTable(
            band=cfg.band,
            bandwidth_mhz=cfg.bandwidth_mhz,
            blocked_channels=set(cfg.blocked_channels),
        )
        self._hop_param2 = (
            cfg.hop_param2 or derive_hop_param2(int.from_bytes(cfg.address, "big"))
        )

        # 功率控制
        self._power_ctrl = PowerController(
            tx_power_dbm=cfg.tx_power_dbm,
            min_power_dbm=cfg.min_power_dbm,
            max_power_dbm=cfg.max_power_dbm,
        )

        # 时序调度
        self._scheduler = ScheduleManager()

        # SMF 调度
        if cfg.smf_enabled:
            self._smf_scheduler = SMFTransmitScheduler()

        # 接入流程
        self._whitelist = AccessWhitelist()
        self._discovery = DiscoveryManager(
            local_address=cfg.address,
            whitelist=self._whitelist,
        )
        self._broadcast_filter = BroadcastFilter()

        # 数据链路参数
        if cfg.frame_type == 2:
            self._data_link_params = AsyncDataLinkParams()
        else:
            self._data_link_params = SyncDataLinkParams()

        # 信道模型 (仿真模式)
        if cfg.transport == TransportMode.SIMULATION and cfg.channel_config is not None:
            self._channel = ChannelModel(config=cfg.channel_config)

        # SDR 硬件
        if cfg.transport == TransportMode.USRP:
            if cfg.sdr_config is not None:
                self._sdr_device = create_device(cfg.sdr_config)
            elif cfg.usrp_config is not None:
                # 向后兼容: 旧版 USRPConfig -> SLETransceiver
                device = USRPDevice(config=cfg.usrp_config, use_mock=True)
                self._transceiver = SLETransceiver(device)

    def _ctrl_bits_len(self) -> int:
        ft = self.config.frame_type
        if ft == 1:
            return 20
        if ft == 2:
            return 28
        return 27

    # ── 生命周期管理 ──

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def link_state(self) -> LinkState:
        return self._link_mgr.state

    @property
    def role(self) -> Role:
        return self._link_mgr.role

    @property
    def peer_address(self) -> bytes:
        return self._peer_address

    @property
    def is_connected(self) -> bool:
        return self._state in (NodeState.CONNECTED, NodeState.PAIRED)

    def start_advertising(self) -> BroadcastFrame | None:
        """进入广播态, 构建并返回扩展广播帧。"""
        self._link_mgr.process_event(Event(EventType.START_BROADCAST))
        self._set_state(NodeState.ADVERTISING)

        self._broadcaster_mgr = BroadcasterAccessManager(
            link_manager=self._link_mgr,
            local_address=self.config.address,
            whitelist=self._whitelist,
        )
        return self._broadcaster_mgr.build_ext_adv_frame()

    def start_scanning(self) -> None:
        """进入扫描态, 搜索广播节点。"""
        self._link_mgr.process_event(Event(EventType.START_SCAN))
        self._set_state(NodeState.SCANNING)
        self._initiator_mgr = InitiatorAccessManager(
            link_manager=self._link_mgr,
            local_address=self.config.address,
            whitelist=self._whitelist,
        )

    def on_broadcast_received(self, frame: BroadcastFrame) -> bool:
        """处理收到的广播帧。

        在扫描态时, 通过发现管理器和广播过滤器处理帧。

        :returns: True 表示该帧包含有效的接入资源配置。
        """
        if not self._broadcast_filter.match(frame):
            return False

        self._discovery.on_broadcast_received(frame)
        self.callback.on_broadcast_received(frame)

        if self._initiator_mgr is not None:
            return self._initiator_mgr.process_ext_adv(frame)
        return False

    def connect(self, peer_address: bytes) -> None:
        """发起接入请求。"""
        self._peer_address = peer_address
        self._link_mgr.process_event(Event(EventType.SEND_ACCESS_REQUEST))
        self._set_state(NodeState.CONNECTING)

    def accept_connection(self, peer_address: bytes, role: Role = Role.G_NODE) -> None:
        """接受接入请求, 直接进入链接态。

        接入完成后自动注册链路调度资源。
        """
        self._peer_address = peer_address
        self._link_mgr.peer_address = peer_address
        self._link_mgr.process_event(
            Event(EventType.ACCESS_REQUEST_RECEIVED, {"accepted": True, "role": role})
        )
        self._set_state(NodeState.CONNECTED)
        self._register_link_schedule(link_id=0)

    def run_access(
        self,
        peer_address: bytes,
        is_broadcaster: bool = True,
    ) -> tuple[BroadcasterAccessManager, InitiatorAccessManager]:
        """执行完整接入编排流程。"""
        b_mgr, i_mgr = run_access_procedure(
            broadcaster_addr=self.config.address if is_broadcaster else peer_address,
            initiator_addr=peer_address if is_broadcaster else self.config.address,
        )
        self._peer_address = peer_address
        self._set_state(NodeState.CONNECTED)
        self._link_mgr.process_event(
            Event(EventType.ACCESS_REQUEST_RECEIVED, {"accepted": True, "role": Role.G_NODE})
        )
        self._register_link_schedule(link_id=0)
        return b_mgr, i_mgr

    def disconnect(self) -> None:
        """断开连接。"""
        self._link_mgr.process_event(Event(EventType.DISCONNECT_REQUEST))
        self._set_state(NodeState.DISCONNECTED)
        self._cleanup()

    def reset(self) -> None:
        """重置到初始状态。"""
        if self._link_mgr.state in (LinkState.CONNECTED, LinkState.PAIRING):
            self._link_mgr.process_event(Event(EventType.DISCONNECT_REQUEST))
        if self._link_mgr.state != LinkState.IDLE:
            self._link_mgr.reset()
        self._set_state(NodeState.IDLE)
        self._cleanup()
        self.__post_init__()

    # ── 安全配对 ──

    def start_pairing(self, peer_address: bytes) -> list:
        """发起配对流程。

        :returns: 待发送给对端的配对信令列表。
        """
        is_g = self._link_mgr.role == Role.G_NODE
        self._pairing = PairingManager(
            is_g_node=is_g,
            local_address=self.config.address,
            peer_address=peer_address,
        )
        self._link_mgr.process_event(Event(EventType.START_PAIRING))
        self._set_state(NodeState.PAIRED)
        return self._pairing.start_pairing()

    def process_pairing_message(self, msg: object) -> list:
        """处理收到的配对信令。

        :returns: 待发送的响应信令列表。
        """
        if self._pairing is None:
            return []
        responses = self._pairing.process_message(msg)
        if self._pairing.state == PairingState.COMPLETED:
            self._setup_crypto()
            self._link_mgr.process_event(Event(EventType.PAIRING_COMPLETE))
            self._set_state(NodeState.CONNECTED)
        elif self._pairing.state == PairingState.FAILED:
            self._link_mgr.process_event(Event(EventType.PAIRING_FAILED))
            self._set_state(NodeState.DISCONNECTED)
        return responses

    def _setup_crypto(self) -> None:
        if self._pairing is None or not self._pairing.is_paired:
            return
        is_g = self._link_mgr.role == Role.G_NODE
        ik = self._pairing.integrity_key
        iv_base = ik[:8] if ik else self._pairing.session_key[:8]
        self._crypto = FrameCryptoContext(
            session_key=self._pairing.session_key,
            iv_base=iv_base,
            direction=0 if is_g else 1,
            mic_len=4,
            frame_type=self.config.frame_type,
            link_id=0,
        )

    # ── 数据发送 ──

    def send(self, data: bytes, priority: Priority = Priority.NORMAL) -> bool:
        """提交数据到发送队列。"""
        return self._qos.submit_data(data, priority)

    def transmit(self) -> TxResult:
        """从发送队列取出数据, 构建帧并生成 IQ 信号。

        自动选择当前跳频信道, 更新发射功率。
        """
        decision, item = self._qos.prepare_tx()
        if item is None:
            return TxResult(iq=None, mac_bytes=None, decision=decision)

        payload = item.data
        encrypted = False

        if self._crypto is not None and self.config.enable_encryption:
            aad = b""
            ciphertext, mic = self._crypto.encrypt(payload, aad)
            payload = ciphertext + mic
            encrypted = True

        frame = AsyncDataFrame(segment_type=0, data=payload)
        mac_bytes = frame.pack()

        ctrl_fields = self._qos.get_ctrl_fields()
        ctrl_info = _build_ctrl_info(ctrl_fields, len(mac_bytes))
        iq = mac_to_iq(mac_bytes, self._tx_config, ctrl_info=ctrl_info)

        # 跳频: 计算当前信道
        slot = self._scheduler.slot_counter.value
        channel = data_link_hop(slot, self._hop_param2, self._freq_table)

        # SDR/USRP 模式: 通过硬件发送
        if self._sdr_device is not None:
            try:
                self._sdr_device.transmit(iq)
            except (OSError, RuntimeError):
                log.warning("SDR 发送失败")
        elif self._transceiver is not None:
            try:
                self._transceiver.transmit_iq(iq)
            except (OSError, RuntimeError):
                log.warning("USRP 发送失败")

        self._tx_count += 1
        return TxResult(
            iq=iq, mac_bytes=mac_bytes, decision=decision,
            encrypted=encrypted, channel=channel,
        )

    def receive(self, iq_signal: np.ndarray, n_mac_bytes: int) -> RxResult:
        """从 IQ 信号解码数据。

        如果配置了信道模型, 会在解码前应用信道效应。
        """
        # 信道模型
        if self._channel is not None:
            iq_signal = self._channel.apply_fading(iq_signal, self._tx_config.sps)

        rx: MacRxResult = iq_to_mac(iq_signal, self._tx_config, n_mac_bytes)

        try:
            recovered = AsyncDataFrame.unpack(rx.mac_payload)
            data = recovered.data
        except (ValueError, IndexError):
            data = None

        if not rx.crc_ok:
            # CRC 失败但帧结构完整: 仍返回数据, 供 Byte BER 逐字节比对
            return RxResult(data=data, success=False)

        data = recovered.data
        decrypted = False

        if self._crypto is not None and self.config.enable_encryption:
            mic_len = self._crypto.mic_len
            if len(data) < mic_len:
                return RxResult(data=None, success=False)
            ciphertext = data[:-mic_len]
            mic = data[-mic_len:]
            try:
                data = self._crypto.decrypt(ciphertext, mic, b"")
                decrypted = True
            except (ValueError, InvalidTag):
                return RxResult(data=None, success=False)

        self._rx_count += 1
        self.callback.on_data_received(data)
        return RxResult(data=data, success=True, decrypted=decrypted)

    def process_feedback(self, crc_ok: bool) -> TxDecision:
        """处理对端 ACK/NACK 反馈。"""
        return self._qos.on_tx_feedback(crc_ok)

    # ── 跳频 ──

    @property
    def freq_table(self) -> FreqTable:
        return self._freq_table

    @property
    def current_channel(self) -> int:
        """当前跳频信道号。"""
        return data_link_hop(
            self._scheduler.slot_counter.value,
            self._hop_param2,
            self._freq_table,
        )

    def block_channel(self, channel: int) -> None:
        """阻塞指定信道。"""
        self._freq_table.blocked_channels.add(channel)

    def unblock_channel(self, channel: int) -> None:
        """解除信道阻塞。"""
        self._freq_table.blocked_channels.discard(channel)

    # ── 功率控制 ──

    @property
    def power_controller(self) -> PowerController:
        return self._power_ctrl

    @property
    def tx_power_dbm(self) -> float:
        return self._power_ctrl.tx_power_dbm

    def adjust_power(self, delta_db: float) -> float:
        """调整发射功率, 返回调整后的功率值。"""
        req = self._power_ctrl.create_request(int(delta_db))
        resp = self._power_ctrl.handle_request(req)
        self._power_ctrl.apply_response(resp)
        return self._power_ctrl.tx_power_dbm

    # ── 时序调度 ──

    @property
    def scheduler(self) -> ScheduleManager:
        return self._scheduler

    def advance_slot(self, n: int = 1) -> int:
        """推进时隙计数器。"""
        return self._scheduler.advance_time(n)

    def _register_link_schedule(self, link_id: int = 0) -> None:
        """接入完成后注册默认链路调度资源。"""
        timing = EventTimingParams()
        self._scheduler.register_link(link_id, timing)

    # ── SMF ──

    @property
    def smf_scheduler(self) -> SMFTransmitScheduler | None:
        return self._smf_scheduler

    def configure_smf(self, params: SMFScheduleParams) -> None:
        """配置 SMF 调度参数。"""
        if self._smf_scheduler is not None:
            self._smf_scheduler.configure(params)
            smf_config = SmfScheduleConfig(
                smf_interval=params.interval,
                frame_type=params.frame_type,
                bandwidth=params.bandwidth,
            )
            self._scheduler.configure_smf(smf_config)

    # ── 接入白名单与发现 ──

    @property
    def whitelist(self) -> AccessWhitelist:
        return self._whitelist

    @property
    def discovery(self) -> DiscoveryManager:
        return self._discovery

    @property
    def discovered_devices(self) -> dict:
        return self._discovery.discovered_devices

    def set_broadcast_filter(self, broadcast_filter: BroadcastFilter) -> None:
        """设置广播帧过滤器。"""
        self._broadcast_filter = broadcast_filter

    # ── 非链接态广播 ──

    def start_non_connected_broadcast(
        self,
        nc_config: NonConnectedBroadcastConfig | None = None,
    ) -> BroadcastFrame | None:
        """发送非链接态广播数据帧。"""
        if nc_config is None:
            nc_config = NonConnectedBroadcastConfig()
        mgr = NonConnectedBroadcastManager(
            config=nc_config,
            local_address=self.config.address,
        )
        return mgr.build_non_connected_broadcast_frame()

    # ── 信道模型 ──

    @property
    def channel_model(self) -> ChannelModel | None:
        return self._channel

    def set_channel_model(self, config: ChannelConfig) -> None:
        """配置仿真信道模型。"""
        self._channel = ChannelModel(config=config)

    def clear_channel_model(self) -> None:
        """清除信道模型 (恢复理想信道)。"""
        self._channel = None

    # ── SDR 硬件 ──

    @property
    def sdr_device(self) -> SDRDevice | None:
        return self._sdr_device

    @property
    def transceiver(self) -> SLETransceiver | None:
        return self._transceiver

    def open_transceiver(self, rx_buf_size: int = 4096) -> None:
        """打开 SDR/USRP 硬件收发器。"""
        if self._transceiver is not None:
            self._transceiver.open(rx_buf_size)

    def close_transceiver(self) -> None:
        """关闭 SDR/USRP 硬件收发器。"""
        if self._sdr_device is not None:
            self._sdr_device.close()
        if self._transceiver is not None:
            self._transceiver.close()

    def receive_iq(self, num_samps: int) -> np.ndarray | None:
        """从 SDR/USRP 接收 IQ 采样。"""
        if self._sdr_device is not None:
            return self._sdr_device.receive(num_samps)
        if self._transceiver is None:
            return None
        return self._transceiver.receive_iq(num_samps)

    # ── 测量 ──

    def generate_measurement_signal(
        self, n_measur: int = 64, security_type: int = 1,
    ) -> np.ndarray:
        """生成窄带测量信号 (标准 6.2.4)。"""
        return measurement_signal_1(n_measur, security_type)

    # ── 数据链路参数 ──

    @property
    def data_link_params(self) -> AsyncDataLinkParams | SyncDataLinkParams | None:
        return self._data_link_params

    # ── 信令 ──

    def send_signaling(self, msg: object) -> ControlFrame | None:
        """通过链路管理器发送控制面信令。"""
        if not self.is_connected:
            return None
        return self._link_mgr.send_signaling(msg)

    def receive_signaling(self, msg: object) -> None:
        """处理收到的控制面信令。"""
        self._link_mgr.process_event(Event(EventType.SIGNALING_RECEIVED, msg))

    # ── 状态查询 ──

    @property
    def stats(self) -> dict:
        """返回节点统计信息。"""
        qos_fields = self._qos.get_ctrl_fields()
        return {
            "state": self._state.name,
            "role": self._link_mgr.role.name,
            "tx_count": self._tx_count,
            "rx_count": self._rx_count,
            "tx_sn": qos_fields.get("tx_sn", 0),
            "rx_sn": qos_fields.get("rx_sn", 0),
            "fer": self._qos.quality.fer,
            "mcs": self.config.mcs_index,
            "queue_size": self._qos.tx_queue.size,
            "flow_paused": self._qos.flow.is_paused,
            "paired": self._pairing is not None and self._pairing.is_paired,
            "encrypted": self._crypto is not None,
            "channel": self.current_channel,
            "tx_power_dbm": self._power_ctrl.tx_power_dbm,
            "hop_param2": self._hop_param2,
        }

    @property
    def recommended_mcs(self) -> int:
        """基于链路质量建议的 MCS 索引。"""
        adj = self._qos.quality.suggest_mcs_adjustment()
        return max(0, min(12, self.config.mcs_index + adj))

    def update_mcs(self, mcs_index: int) -> None:
        """更新 MCS 索引。"""
        self.config.mcs_index = max(0, min(12, mcs_index))
        self._tx_config = TxConfig(
            frame_type=self.config.frame_type,
            mcs_index=self.config.mcs_index,
            pid=self._tx_config.pid,
            whitening_seed=self._tx_config.whitening_seed,
            crc_seed=self._tx_config.crc_seed,
            crc_len=self._tx_config.crc_len,
            ctrl_bits_len=self._tx_config.ctrl_bits_len,
            pilot_interval=self._tx_config.pilot_interval,
        )

    # ── 内部方法 ──

    def _set_state(self, new: NodeState) -> None:
        old = self._state
        if old != new:
            self._state = new
            self.callback.on_state_changed(old, new)

    def _cleanup(self) -> None:
        self._peer_address = b""
        self._pairing = None
        self._crypto = None
        self._broadcaster_mgr = None
        self._initiator_mgr = None


class _InternalCallback(LinkManagerCallback):
    """将 LinkManager 回调转发到 SleNode。"""

    def __init__(self, node: SleNode) -> None:
        self._node = node

    def on_access_accepted(self, role: Role) -> None:
        self._node.callback.on_connected(self._node.peer_address, role)

    def on_disconnected(self, reason: DisconnectReason) -> None:
        self._node._set_state(NodeState.DISCONNECTED)
        self._node.callback.on_disconnected(reason)


def _build_ctrl_info(
    ctrl_fields: dict[str, int], data_length: int,
) -> ControlInfoA2:
    """从 QoS 控制字段构建 A2 控制信息。"""
    return ControlInfoA2(
        packet_type=0,
        empty_packet=ctrl_fields.get("empty_packet", 0) & 1,
        tx_sn=ctrl_fields.get("tx_sn", 0) & 1,
        rx_sn=ctrl_fields.get("rx_sn", 0) & 1,
        flow_ctrl=ctrl_fields.get("flow_ctrl", 0) & 1,
        sys_mgmt_rx=0,
        reserved=0,
        data_length=data_length,
    )


def _build_ctrl_bits(
    ctrl_fields: dict[str, int], frame_type: int,
) -> np.ndarray:
    """从 QoS 控制字段构建控制信息比特。"""
    if frame_type == 2:
        bits = np.zeros(28, dtype=np.int8)
        bits[3] = ctrl_fields.get("tx_sn", 0) & 1
        bits[4] = ctrl_fields.get("rx_sn", 0) & 1
        bits[5] = ctrl_fields.get("flow_ctrl", 0) & 1
    elif frame_type in (3, 4):
        bits = np.zeros(27, dtype=np.int8)
        tx_sn = ctrl_fields.get("tx_sn", 0) & 0x1F
        for i in range(5):
            bits[3 + i] = (tx_sn >> (4 - i)) & 1
        rx_sn = ctrl_fields.get("rx_sn", 0) & 0x1F
        for i in range(5):
            bits[8 + i] = (rx_sn >> (4 - i)) & 1
        bits[13] = ctrl_fields.get("flow_ctrl", 0) & 1
    else:
        bits = np.zeros(20, dtype=np.int8)
    return bits
