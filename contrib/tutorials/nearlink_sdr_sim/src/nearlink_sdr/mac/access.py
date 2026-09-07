"""接入流程管理 -- TXS-10002-2025 标准 7.1.3。

实现 SLE 设备发现和接入的完整六阶段流程:
  a) 广播方准备并发送可接入扩展广播帧
  b) 接入发起方发送接入请求帧
  c) 广播方接收请求并发送响应
  d) 接入方接收响应并进入链接态
  e) 数据链路建立
  f) 安全流程 (委托给 security 模块)

协调 BroadcastFrame、AccessBasicInfo、TransportIndicationInfo、
AccessRequestInfo、AccessResponseInfo 等数据结构完成端到端接入。
"""

from __future__ import annotations

__all__ = [
    "MAX_ADV_INTERVAL_US",
    "MAX_ADV_RANDOM_DELAY_US",
    "MIN_ADV_INTERVAL_US",
    "MIN_ADV_TO_EXT_ADV_GAP",
    "MIN_EXT_ADV_TO_REQUEST_GAP",
    "MIN_REQUEST_TO_RESPONSE_GAP",
    "AccessConfig",
    "AccessPhase",
    "AccessWhitelist",
    "BroadcasterAccessManager",
    "DiscoveryManager",
    "InitiatorAccessManager",
    "NegotiatedRole",
    "NonConnectedBroadcastConfig",
    "NonConnectedBroadcastManager",
    "NonConnectedBroadcastResult",
    "negotiate_gt_role",
    "parse_non_connected_broadcast",
    "run_access_procedure",
]


import logging
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from nearlink_sdr.mac.broadcast import (
    AccessBasicInfo,
    AccessRequestInfo,
    AccessResponseEntry,
    AccessResponseInfo,
    AccessResponseType,
    BroadcastDataType,
    BroadcastFrame,
    DiscoveryAccessEntry,
    DiscoveryAccessResourceConfig,
    GTNegotiation,
    NonLinkedBroadcastLinkInfo,
    QueryRequestFilterInfo,
    SystemMgmtFrameInfo,
    TransportIndicationInfo,
)
from nearlink_sdr.mac.link_manager import (
    Event,
    EventType,
    LinkManager,
    Role,
)

log = logging.getLogger(__name__)

# ── 最小时间间隔 (标准 7.1.3, 单位 μs) ──
MIN_ADV_TO_EXT_ADV_GAP = 300
MIN_EXT_ADV_TO_REQUEST_GAP = 300
MIN_REQUEST_TO_RESPONSE_GAP = 300

# ── 广播间隔约束 (标准 7.1.1, 单位 μs) ──
MIN_ADV_INTERVAL_US = 4_000              # 4 ms
MAX_ADV_INTERVAL_US = 2_097_151_875      # 2097.151875 s
MAX_ADV_RANDOM_DELAY_US = 2_000          # 2 ms


class AccessPhase(IntEnum):
    """接入流程阶段标识。"""
    IDLE = 0
    ADV_SENDING = 1        # a: 广播方发送扩展广播帧
    REQ_WINDOW = 2         # b: 接入请求窗口
    RSP_WINDOW = 3         # c: 接入响应窗口
    LINK_SETUP = 4         # d/e: 链路建立
    COMPLETED = 5          # 接入完成


@dataclass
class AccessConfig:
    """接入流程配置参数。"""
    # 广播/接入资源配置
    request_offset_us: int = 600       # 扩展广播帧结束到请求窗口 (μs)
    request_max_length: int = 64       # 接入请求帧最大数据长度 (字节)
    response_offset_us: int = 1200     # 扩展广播帧结束到响应窗口 (μs)
    window_count: int = 1              # 并行请求窗口个数
    # GT 角色协商
    gt_preference: int = 0             # 0=偏好T节点, 1=偏好G节点
    gt_negotiable: bool = True         # 角色是否可协商
    # 超时与重试
    access_timeout_ms: int = 5000      # 接入超时
    max_retries: int = 3               # 最大重试次数


@dataclass
class NegotiatedRole:
    """GT 角色协商结果。"""
    local_role: Role
    peer_role: Role
    negotiated: bool = False           # 是否经过协商 (vs 默认分配)


# ---------------------------------------------------------------------------
# 7.1.6 接入白名单
# ---------------------------------------------------------------------------


@dataclass
class AccessWhitelist:
    """接入白名单 (7.1.6)

    广播设备设置白名单后, 只接收白名单中设备的接入请求。
    接入设备设置白名单后, 只向白名单中设备发起接入。
    """
    _addresses: set[bytes] = field(default_factory=set)
    enabled: bool = False

    def add(self, address: bytes) -> None:
        """添加一个设备地址 (6 字节)"""
        self._addresses.add(bytes(address[:6]))

    def remove(self, address: bytes) -> None:
        """移除一个设备地址"""
        self._addresses.discard(bytes(address[:6]))

    def clear(self) -> None:
        """清空白名单"""
        self._addresses.clear()

    def contains(self, address: bytes) -> bool:
        """检查地址是否在白名单中"""
        return bytes(address[:6]) in self._addresses

    def check(self, address: bytes) -> bool:
        """检查是否允许该地址的接入

        白名单未启用时允许所有地址; 启用后仅允许白名单内地址。
        """
        if not self.enabled:
            return True
        return self.contains(address)

    @property
    def addresses(self) -> list[bytes]:
        return sorted(self._addresses)


# ---------------------------------------------------------------------------
# 7.1.2 发现流程
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryManager:
    """发现流程管理器 (标准 7.1.2)

    实现发现设备接收广播帧后的查询请求/响应流程:
      a) 广播设备发送基础广播帧和扩展广播帧
      b) 发现设备收到可查询扩展广播帧后发送查询请求帧
      c) 广播设备接收请求并发送查询响应帧
      d) 发现设备接收查询响应帧, 完成发现
    """
    local_address: bytes = b"\x00" * 6
    whitelist: AccessWhitelist = field(default_factory=AccessWhitelist)
    # 发现到的设备列表: {地址: (广播帧, 查询响应帧)}
    _discovered: dict[bytes, tuple[BroadcastFrame, BroadcastFrame | None]] = (
        field(default_factory=dict)
    )
    # 广播方: 查询请求处理器的数据存储
    _query_responses: dict[bytes, BroadcastFrame] = field(default_factory=dict)

    # -- 发现设备端 --

    def on_broadcast_received(self, frame: BroadcastFrame) -> bool:
        """处理收到的广播帧。

        若帧中包含发现/接入资源配置且为可查询帧, 返回 True 表示可发送查询请求。
        白名单启用时仅处理白名单内设备。

        :param frame: 收到的广播帧。

        :returns: True 表示可查询, 需要后续发送查询请求。
        """
        if not self.whitelist.check(frame.local_addr):
            return False

        addr = bytes(frame.local_addr)
        self._discovered[addr] = (frame, None)

        # 检查是否包含可查询的发现接入资源配置
        for data_type, data_bytes in frame.data_items:
            if data_type == BroadcastDataType.DISCOVERY_ACCESS_RESOURCE:
                config = DiscoveryAccessResourceConfig.unpack(data_bytes)
                if config.entries:
                    for entry in config.entries:
                        if entry.request_type == 1:  # 可查询
                            return True
        return False

    def build_query_request(
        self,
        target_addr: bytes,
        filter_info: QueryRequestFilterInfo | None = None,
        upper_layer_data: bytes = b"",
    ) -> BroadcastFrame:
        """构造查询请求帧 (阶段 b)。

        :param target_addr: 目标广播设备地址。
        :param filter_info: 查询过滤信息 (按服务UUID过滤)。
        :param upper_layer_data: 高层广播数据。

        :returns: 查询请求广播帧。
        """
        data_items: list[tuple[int, bytes]] = []
        if filter_info is not None:
            data_items.append((
                BroadcastDataType.QUERY_REQUEST_FILTER,
                filter_info.pack(),
            ))
        if upper_layer_data:
            data_items.append((
                BroadcastDataType.UPPER_LAYER_DATA,
                upper_layer_data,
            ))

        return BroadcastFrame(
            structure_indication=0x10,
            local_addr_type=0,
            peer_addr_type=0,
            local_addr=self.local_address,
            irk_id=0,
            peer_addr=target_addr,
            data_items=data_items,
        )

    def handle_query_response(
        self, frame: BroadcastFrame,
    ) -> bool:
        """处理查询响应帧 (阶段 d)。

        :param frame: 收到的查询响应帧。

        :returns: True 表示发现完成。
        """
        addr = bytes(frame.local_addr)
        if addr in self._discovered:
            original, _ = self._discovered[addr]
            self._discovered[addr] = (original, frame)
            return True
        return False

    # -- 广播设备端 --

    def handle_query_request(
        self,
        request_frame: BroadcastFrame,
        all_services_data: bytes = b"",
    ) -> BroadcastFrame:
        """处理查询请求并构造查询响应帧 (阶段 c)。

        :param request_frame: 收到的查询请求帧。
        :param all_services_data: 本设备支持的所有服务数据。

        :returns: 查询响应广播帧。
        """
        requester_addr = bytes(request_frame.local_addr)

        # 判断是否携带查询过滤信息
        response_data = all_services_data
        for data_type, _data_bytes in request_frame.data_items:
            if data_type == BroadcastDataType.QUERY_REQUEST_FILTER:
                # 有过滤信息时, 应仅返回过滤指定的服务数据
                # 此处简化处理: 返回全部数据, 由上层做更精确过滤
                break

        data_items: list[tuple[int, bytes]] = []
        if response_data:
            data_items.append((
                BroadcastDataType.UPPER_LAYER_DATA, response_data,
            ))

        response = BroadcastFrame(
            structure_indication=0x10,
            local_addr_type=0,
            peer_addr_type=0,
            local_addr=self.local_address,
            irk_id=0,
            peer_addr=requester_addr,
            data_items=data_items,
        )
        self._query_responses[requester_addr] = response
        return response

    @property
    def discovered_devices(self) -> dict[bytes, tuple[BroadcastFrame, BroadcastFrame | None]]:
        """返回已发现的设备及其广播帧/查询响应帧。"""
        return dict(self._discovered)

    def clear(self) -> None:
        """清空发现结果。"""
        self._discovered.clear()
        self._query_responses.clear()


def negotiate_gt_role(
    broadcaster_pref: int,
    broadcaster_negotiable: bool,
    initiator_pref: int,
    initiator_negotiable: bool,
) -> NegotiatedRole:
    """根据标准 7.1.3 执行 GT 角色协商。

    双方各表达角色偏好 (0=T节点, 1=G节点) 和可协商标志。
    冲突时默认: 发起方→G节点, 广播方→T节点。

    :param broadcaster_pref: 广播方角色偏好 (0=T, 1=G)。
    :param broadcaster_negotiable: 广播方角色是否可协商。
    :param initiator_pref: 发起方 (接入方) 角色偏好。
    :param initiator_negotiable: 发起方角色是否可协商。

    :returns: 协商结果, 从发起方视角: local_role 为发起方角色。
    """
    # 若双方偏好一致 (都想当 G 或都想当 T), 存在冲突
    if broadcaster_pref == initiator_pref:
        # 均不可协商 → 默认分配: 发起方=G, 广播方=T
        if not broadcaster_negotiable and not initiator_negotiable:
            return NegotiatedRole(
                local_role=Role.G_NODE,
                peer_role=Role.T_NODE,
                negotiated=False,
            )
        # 广播方可协商 → 发起方保持偏好
        if broadcaster_negotiable:
            if initiator_pref == 1:
                return NegotiatedRole(Role.G_NODE, Role.T_NODE, True)
            return NegotiatedRole(Role.T_NODE, Role.G_NODE, True)
        # 发起方可协商 → 广播方保持偏好
        if broadcaster_pref == 1:
            return NegotiatedRole(Role.T_NODE, Role.G_NODE, True)
        return NegotiatedRole(Role.G_NODE, Role.T_NODE, True)

    # 偏好互补: 一方想 G, 另一方想 T → 各取所好
    if initiator_pref == 1:
        return NegotiatedRole(Role.G_NODE, Role.T_NODE, True)
    return NegotiatedRole(Role.T_NODE, Role.G_NODE, True)


# ── 广播方接入管理 ──


@dataclass
class BroadcasterAccessManager:
    """广播方接入管理器 (标准 7.1.3 阶段 a/c)。

    职责:
    - 构造可接入扩展广播帧
    - 处理收到的接入请求
    - 生成接入响应
    - 完成 GT 角色协商
    """
    config: AccessConfig = field(default_factory=AccessConfig)
    link_manager: LinkManager = field(default_factory=LinkManager)
    local_address: bytes = b"\x00" * 6
    # 链路配置 (用于 AccessBasicInfo 或 TransportIndicationInfo)
    use_smf: bool = True
    smf_baseline_slot: int = 0
    smf_offset: int = 300
    smf_link_id: int = 0x00000001
    smf_period: int = 800
    smf_frame_type: int = 2
    smf_bandwidth: int = 0
    smf_pilot_density: int = 0
    access_link_id: int = 0x000001
    access_period: int = 40
    access_timeout: int = 50
    sleep_clock_accuracy: int = 7
    access_crc_type: int = 0
    access_crc_init: int = 0
    hop_map: bytes = b"\xFF" * 10
    smf_channel_table: bytes = b"\x00\x01\x02"
    whitelist: AccessWhitelist = field(default_factory=AccessWhitelist)
    # 广播间隔 (标准 7.1.1, 固定时间间隔, 单位 μs)
    adv_interval_us: int = 100_000  # 默认 100 ms
    # 内部状态
    _phase: AccessPhase = AccessPhase.IDLE
    _pending_requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not MIN_ADV_INTERVAL_US <= self.adv_interval_us <= MAX_ADV_INTERVAL_US:
            raise ValueError(
                f"adv_interval_us={self.adv_interval_us} 超出标准范围 "
                f"[{MIN_ADV_INTERVAL_US}, {MAX_ADV_INTERVAL_US}] (§7.1.1)"
            )

    def next_adv_delay_us(self) -> int:
        """计算下一次广播的总间隔 (§7.1.1)。

        返回固定间隔加 [0, 2ms] 随机延迟, 单位 μs。
        """
        return self.adv_interval_us + random.randint(0, MAX_ADV_RANDOM_DELAY_US)

    def build_ext_adv_frame(self) -> BroadcastFrame:
        """构建可接入扩展广播帧 (阶段 a)。

        包含发现接入资源配置信息 (7.1.4.2)。

        :returns: 填充好的 BroadcastFrame 对象。
        """
        gt_neg = GTNegotiation.NEGOTIATE_G
        if self.config.gt_preference == 1:
            if self.config.gt_negotiable:
                gt_neg = GTNegotiation.NEGOTIATE_G
            else:
                gt_neg = GTNegotiation.FIXED_G
        else:
            if self.config.gt_negotiable:
                gt_neg = GTNegotiation.NEGOTIATE_T
            else:
                gt_neg = GTNegotiation.FIXED_T

        discovery_config = DiscoveryAccessResourceConfig(
            request_offset=self.config.request_offset_us,
            request_max_length=self.config.request_max_length,
            response_offset=self.config.response_offset_us,
            gt_negotiation=gt_neg,
            entry_count=self.config.window_count,
            entries=[
                DiscoveryAccessEntry(
                    request_type=1,
                    carry_info_indication=0,
                    peer_addr_type=0,
                    addr_present=0,
                    peer_addr=b"",
                )
                for _ in range(self.config.window_count)
            ],
        )

        frame = BroadcastFrame(
            structure_indication=0x11,
            local_addr_type=0,
            peer_addr_type=0,
            local_addr=self.local_address,
            irk_id=0,
            peer_addr=b"\x00" * 6,
            data_items=[
                (BroadcastDataType.DISCOVERY_ACCESS_RESOURCE,
                 discovery_config.pack()),
            ],
        )
        self._phase = AccessPhase.ADV_SENDING
        return frame

    def handle_access_request(
        self,
        request_data: bytes,
        peer_address: bytes = b"\x00" * 6,
    ) -> tuple[BroadcastFrame | None, bool]:
        """处理接入请求 (阶段 c)。

        :param request_data: 接入请求帧数据。
        :param peer_address: 请求方 MAC 地址。

        :returns: (响应帧, 是否接受)。响应帧包含 AccessResponseInfo,
            以及 (若接受且角色为 G) AccessBasicInfo 或 TransportIndicationInfo。
        """
        self._phase = AccessPhase.RSP_WINDOW

        # 白名单检查 (7.1.6)
        if not self.whitelist.check(peer_address):
            log.info("接入请求被白名单拒绝: %s", peer_address.hex())
            return None, False

        # 解析接入请求
        req = AccessRequestInfo.unpack(request_data)

        # GT 角色协商
        initiator_gt_pref = 0
        initiator_negotiable = True
        if req.gt_role is not None:
            initiator_gt_pref = req.gt_role & 0x01
            initiator_negotiable = (req.gt_role & 0x02) == 0

        negotiation = negotiate_gt_role(
            broadcaster_pref=self.config.gt_preference,
            broadcaster_negotiable=self.config.gt_negotiable,
            initiator_pref=initiator_gt_pref,
            initiator_negotiable=initiator_negotiable,
        )

        # 从发起方视角看, negotiation.local_role 是发起方角色
        # 广播方角色 = negotiation.peer_role
        broadcaster_role = negotiation.peer_role

        # 构建响应
        response_entry = AccessResponseEntry(
            peer_addr=peer_address,
            response_type=AccessResponseType.ACCEPT,
            peer_addr_type=0,
            repeat_indication=0,
        )
        response_info = AccessResponseInfo(
            entries=[response_entry],
        )

        data_items: list[tuple[int, bytes]] = [
            (BroadcastDataType.ACCESS_RESPONSE, response_info.pack()),
        ]

        # 若广播方成为 G 节点, 需在响应中携带链路配置
        if broadcaster_role == Role.G_NODE and self.use_smf:
                access_basic = AccessBasicInfo(
                    smf_baseline_slot=self.smf_baseline_slot,
                    smf_offset=self.smf_offset,
                    smf_link_id=self.smf_link_id,
                    smf_period=self.smf_period,
                    smf_frame_type=self.smf_frame_type,
                    smf_bandwidth=self.smf_bandwidth,
                    smf_pilot_density=self.smf_pilot_density,
                    access_link_id=self.access_link_id,
                    access_period=self.access_period,
                    access_timeout=self.access_timeout,
                    sleep_clock_accuracy=self.sleep_clock_accuracy,
                    access_crc_type=self.access_crc_type,
                    access_crc_init=self.access_crc_init,
                    hop_map=self.hop_map,
                    smf_channel_count=len(self.smf_channel_table),
                    smf_channel_table=self.smf_channel_table,
                )
                data_items.append(
                    (BroadcastDataType.ACCESS_BASIC, access_basic.pack())
                )

        response_frame = BroadcastFrame(
            structure_indication=0x11,
            local_addr_type=0,
            peer_addr_type=0,
            local_addr=self.local_address,
            irk_id=0,
            peer_addr=b"\x00" * 6,
            data_items=data_items,
        )

        # 驱动状态机
        self.link_manager.local_address = self.local_address
        self.link_manager.process_event(Event(
            EventType.ACCESS_REQUEST_RECEIVED,
            data={
                "accepted": True,
                "role": broadcaster_role,
                "peer_address": peer_address,
            },
        ))

        self._phase = AccessPhase.COMPLETED
        return response_frame, True

    def reject_access_request(
        self,
        peer_address: bytes = b"\x00" * 6,
        reason: AccessResponseType = AccessResponseType.USER_REJECT,
    ) -> BroadcastFrame:
        """拒绝接入请求。

        :param peer_address: 请求方 MAC 地址。
        :param reason: 拒绝原因。

        :returns: 包含拒绝响应的帧。
        """
        response_entry = AccessResponseEntry(
            peer_addr=peer_address,
            response_type=reason,
            peer_addr_type=0,
            repeat_indication=0,
        )
        response_info = AccessResponseInfo(
            entries=[response_entry],
        )
        return BroadcastFrame(
            structure_indication=0x11,
            local_addr_type=0,
            peer_addr_type=0,
            local_addr=self.local_address,
            irk_id=0,
            peer_addr=b"\x00" * 6,
            data_items=[
                (BroadcastDataType.ACCESS_RESPONSE, response_info.pack()),
            ],
        )


# ── 接入发起方管理 ──


@dataclass
class InitiatorAccessManager:
    """接入发起方管理器 (标准 7.1.3 阶段 b/d)。

    职责:
    - 解析收到的扩展广播帧
    - 构造接入请求
    - 处理接入响应
    - 完成 GT 角色协商
    """
    config: AccessConfig = field(default_factory=AccessConfig)
    link_manager: LinkManager = field(default_factory=LinkManager)
    local_address: bytes = b"\x00" * 6
    whitelist: AccessWhitelist = field(default_factory=AccessWhitelist)
    # 解析出的广播方信息
    _adv_frame: BroadcastFrame | None = None
    _discovery_config: DiscoveryAccessResourceConfig | None = None
    _phase: AccessPhase = AccessPhase.IDLE
    _retry_count: int = 0

    def process_ext_adv(self, frame: BroadcastFrame) -> bool:
        """处理收到的可接入扩展广播帧 (阶段 b 准备)。

        解析发现接入资源配置, 提取请求窗口参数。
        白名单启用时, 仅处理白名单中设备的广播帧 (7.1.6)。

        :param frame: 收到的广播帧。

        :returns: True 表示帧中包含有效的接入资源配置。
        """
        # 白名单检查 (7.1.6)
        if not self.whitelist.check(frame.local_addr):
            log.info("广播帧被白名单过滤: %s", frame.local_addr.hex())
            return False

        self._adv_frame = frame
        for data_type, data_bytes in frame.data_items:
            if data_type == BroadcastDataType.DISCOVERY_ACCESS_RESOURCE:
                self._discovery_config = DiscoveryAccessResourceConfig.unpack(
                    data_bytes
                )
                self._phase = AccessPhase.REQ_WINDOW
                # 通知 link_manager
                self.link_manager.process_event(Event(
                    EventType.BROADCAST_RECEIVED, data=frame,
                ))
                return True
        return False

    def build_access_request(self) -> bytes:
        """构造接入请求帧数据 (阶段 b)。

        根据本地角色偏好构建 AccessRequestInfo。

        :returns: 接入请求帧的序列化数据。
        """
        gt_flag = self.config.gt_preference
        if not self.config.gt_negotiable:
            gt_flag |= 0x02

        req = AccessRequestInfo(
            structure_indication=0xFF,  # 所有字段均存在
            gt_role=gt_flag,
            frame_support=0x0F,          # 支持 FT1-4
            bandwidth_support=0x07,      # 支持 1M/2M/4M
            mcs_support=0x1FFF,          # 支持所有 MCS
            pilot_support=0x0F,          # 四种导频密度
            slot_support=0x1F,           # 五种调度时隙
            switch_delay=0x00,           # 125μs
            crc_support=0x03,            # CRC24 + CRC32
        )

        # 驱动状态机
        self.link_manager.process_event(Event(
            EventType.SEND_ACCESS_REQUEST,
            data={
                "peer_address": self._adv_frame.local_addr
                if self._adv_frame else b"",
                "role": Role.G_NODE if self.config.gt_preference == 1
                else Role.T_NODE,
            },
        ))

        return req.pack()

    def handle_access_response(
        self,
        response_data: bytes,
    ) -> tuple[Role | None, dict[str, Any]]:
        """处理接入响应 (阶段 d)。

        :param response_data: 接入响应帧完整数据 (BroadcastFrame.pack() 格式)。

        :returns: (最终角色, 链路参数字典)。角色为 None 表示接入被拒绝。
        """
        frame = BroadcastFrame.unpack(response_data)
        link_params: dict[str, Any] = {}
        response_accepted = False
        final_role: Role | None = None

        for data_type, data_bytes in frame.data_items:
            if data_type == BroadcastDataType.ACCESS_RESPONSE:
                resp = AccessResponseInfo.unpack(data_bytes)
                if resp.entries and resp.entries[0].response_type == \
                        AccessResponseType.ACCEPT:
                    response_accepted = True
                else:
                    response_accepted = False

            elif data_type == BroadcastDataType.ACCESS_BASIC:
                access_info = AccessBasicInfo.unpack(data_bytes)
                link_params["smf_baseline_slot"] = access_info.smf_baseline_slot
                link_params["smf_offset"] = access_info.smf_offset
                link_params["smf_link_id"] = access_info.smf_link_id
                link_params["access_link_id"] = access_info.access_link_id
                link_params["supervision_timeout"] = (
                    access_info.access_timeout * 10
                )
                link_params["crc_type"] = access_info.access_crc_type
                link_params["hop_map"] = access_info.hop_map

            elif data_type == BroadcastDataType.TRANSPORT_INDICATION:
                ti = TransportIndicationInfo.unpack(data_bytes)
                link_params["event_group_period"] = ti.event_group_period
                link_params["event_period"] = ti.event_period

        if response_accepted:
            # 确定最终角色: 响应中有 AccessBasicInfo 说明对端是 G 节点
            if BroadcastDataType.ACCESS_BASIC in \
                    [dt for dt, _ in frame.data_items]:
                final_role = Role.T_NODE  # 对端 G, 本端 T
            else:
                final_role = Role.G_NODE if self.config.gt_preference == 1 \
                    else Role.T_NODE

            self.link_manager.process_event(Event(
                EventType.ACCESS_RESPONSE_RECEIVED,
                data={"accepted": True, "role": final_role},
            ))
            self._phase = AccessPhase.COMPLETED
        else:
            self._retry_count += 1
            self.link_manager.process_event(Event(
                EventType.ACCESS_RESPONSE_RECEIVED,
                data={"accepted": False},
            ))
            self._phase = AccessPhase.IDLE

        return final_role, link_params

    @property
    def can_retry(self) -> bool:
        """是否还能重试接入。"""
        return self._retry_count < self.config.max_retries

    @property
    def discovery_config(self) -> DiscoveryAccessResourceConfig | None:
        return self._discovery_config

    @property
    def phase(self) -> AccessPhase:
        return self._phase


# ── 非链接态广播管理 (7.1.7.2) ──


@dataclass
class NonConnectedBroadcastConfig:
    """非链接态广播传输配置。"""
    # 广播链路信息
    transmission_type: int = 0           # 0=异步, 1=同步
    service_adapt_mode: int = 0          # 0=周期, 1=非周期
    system_slot_seq: int = 0
    event_group_offset: int = 0
    event_group_set_id: int = 0
    event_group_count: int = 1
    event_group_interval: int = 10
    event_group_period: int = 160
    event_period: int = 40
    event_count: int = 1
    sync_anchor_delay: int = 0
    sync_ref_delay: int = 0
    base_link_id: int = 0x000001
    frame_type: int = 2
    bandwidth: int = 0
    pilot_density: int = 0
    sdu_max: int = 128
    sdu_period: int = 1000
    pdu_max: int = 256
    new_packet_count: int = 1
    crc_type: int = 0
    crc_base_init: int = 0
    hop_map: bytes = b"\xFF" * 10
    is_5g: bool = False
    # 系统管理帧信息
    smf_baseline_slot: int = 0
    smf_offset: int = 300
    smf_access_addr: int = 0x000001
    smf_period: int = 800
    smf_frame_type: int = 2
    smf_bandwidth: int = 0
    smf_pilot_density: int = 0
    smf_channel_table: bytes = b"\x00\x01\x02"
    # 加密参数 (可选)
    encrypted: bool = False
    giv: bytes = b""
    gskd: bytes = b""


@dataclass
class NonConnectedBroadcastManager:
    """非链接态广播管理器 (标准 7.1.7.2)。

    通过携带非链接态广播信息和启动系统管理帧信息的扩展广播帧
    建立非链接态广播传输。
    """
    config: NonConnectedBroadcastConfig = field(
        default_factory=NonConnectedBroadcastConfig,
    )
    local_address: bytes = b"\x00" * 6

    def build_non_connected_broadcast_frame(self) -> BroadcastFrame:
        """构建携带非链接态广播信息的扩展广播帧。

        帧中包含两种数据:
        - UNLINKED_BROADCAST_LINK (0x06): NonLinkedBroadcastLinkInfo
        - SYSTEM_MGMT_FRAME (0x05): SystemMgmtFrameInfo

        :returns: 构建好的 BroadcastFrame。
        """
        cfg = self.config

        link_info = NonLinkedBroadcastLinkInfo(
            transmission_type=cfg.transmission_type,
            service_adapt_mode=cfg.service_adapt_mode,
            system_slot_seq=cfg.system_slot_seq,
            event_group_offset=cfg.event_group_offset,
            event_group_set_id=cfg.event_group_set_id,
            event_group_count=cfg.event_group_count,
            event_group_interval=cfg.event_group_interval,
            event_group_period=cfg.event_group_period,
            event_period=cfg.event_period,
            event_count=cfg.event_count,
            sync_anchor_delay=cfg.sync_anchor_delay,
            sync_ref_delay=cfg.sync_ref_delay,
            base_link_id=cfg.base_link_id,
            frame_type=cfg.frame_type,
            bandwidth=cfg.bandwidth,
            pilot_density=cfg.pilot_density,
            sdu_max=cfg.sdu_max,
            sdu_period=cfg.sdu_period,
            pdu_max=cfg.pdu_max,
            new_packet_count=cfg.new_packet_count,
            crc_type=cfg.crc_type,
            crc_base_init=cfg.crc_base_init,
            hop_map=cfg.hop_map,
            giv=cfg.giv if cfg.encrypted else b"",
            gskd=cfg.gskd if cfg.encrypted else b"",
            is_5g=cfg.is_5g,
        )

        smf_info = SystemMgmtFrameInfo(
            baseline_slot=cfg.smf_baseline_slot,
            offset=cfg.smf_offset,
            access_addr=cfg.smf_access_addr,
            period=cfg.smf_period,
            frame_type=cfg.smf_frame_type,
            bandwidth=cfg.smf_bandwidth,
            pilot_density=cfg.smf_pilot_density,
            channel_count=len(cfg.smf_channel_table),
            channel_table=cfg.smf_channel_table,
        )

        data_items: list[tuple[int, bytes]] = [
            (BroadcastDataType.UNLINKED_BROADCAST_LINK, link_info.pack()),
            (BroadcastDataType.SYSTEM_MGMT_FRAME, smf_info.pack()),
        ]

        return BroadcastFrame(
            structure_indication=0x11,
            local_addr_type=0,
            peer_addr_type=0,
            local_addr=self.local_address,
            irk_id=0,
            peer_addr=b"\x00" * 6,
            data_items=data_items,
        )


@dataclass
class NonConnectedBroadcastResult:
    """解析非链接态广播帧的结果。"""
    link_info: NonLinkedBroadcastLinkInfo
    smf_info: SystemMgmtFrameInfo
    broadcaster_addr: bytes


def parse_non_connected_broadcast(
    frame: BroadcastFrame,
) -> NonConnectedBroadcastResult | None:
    """解析扩展广播帧中的非链接态广播信息 (7.1.7.2)。

    :param frame: 收到的扩展广播帧。

    :returns: 解析结果; 若帧中不包含非链接态广播信息则返回 None。
    """
    link_info: NonLinkedBroadcastLinkInfo | None = None
    smf_info: SystemMgmtFrameInfo | None = None

    for data_type, data_bytes in frame.data_items:
        if data_type == BroadcastDataType.UNLINKED_BROADCAST_LINK:
            link_info = NonLinkedBroadcastLinkInfo.unpack(data_bytes)
        elif data_type == BroadcastDataType.SYSTEM_MGMT_FRAME:
            smf_info = SystemMgmtFrameInfo.unpack(data_bytes)

    if link_info is None or smf_info is None:
        return None

    return NonConnectedBroadcastResult(
        link_info=link_info,
        smf_info=smf_info,
        broadcaster_addr=frame.local_addr,
    )


# ── 端到端接入流程 (仿真/测试用) ──


def run_access_procedure(
    broadcaster_addr: bytes = b"\x01\x02\x03\x04\x05\x06",
    initiator_addr: bytes = b"\x0A\x0B\x0C\x0D\x0E\x0F",
    broadcaster_config: AccessConfig | None = None,
    initiator_config: AccessConfig | None = None,
    broadcaster_use_smf: bool = True,
) -> tuple[BroadcasterAccessManager, InitiatorAccessManager]:
    """执行完整的端到端接入流程 (标准 7.1.3 阶段 a-e)。

    用于仿真和集成测试, 不涉及实际射频传输。

    :param broadcaster_addr: 广播方 MAC 地址。
    :param initiator_addr: 发起方 MAC 地址。
    :param broadcaster_config: 广播方配置。
    :param initiator_config: 发起方配置。
    :param broadcaster_use_smf: 是否使用系统管理帧模式。

    :returns: (广播方管理器, 发起方管理器) 二元组,
        两者的 link_manager 在成功时均处于 CONNECTED 状态。
    """
    if broadcaster_config is None:
        broadcaster_config = AccessConfig()
    if initiator_config is None:
        initiator_config = AccessConfig()

    # 初始化广播方: IDLE → BROADCASTING
    b_mgr = BroadcasterAccessManager(
        config=broadcaster_config,
        local_address=broadcaster_addr,
        use_smf=broadcaster_use_smf,
    )
    b_mgr.link_manager.local_address = broadcaster_addr
    b_mgr.link_manager.process_event(Event(EventType.START_BROADCAST))

    # 初始化发起方: IDLE → SCANNING
    i_mgr = InitiatorAccessManager(
        config=initiator_config,
        local_address=initiator_addr,
    )
    i_mgr.link_manager.local_address = initiator_addr
    i_mgr.link_manager.process_event(Event(EventType.START_SCAN))

    # 阶段 a: 广播方构建扩展广播帧
    ext_adv_frame = b_mgr.build_ext_adv_frame()

    # 阶段 b: 发起方收到广播帧, 解析并发送接入请求
    i_mgr.process_ext_adv(ext_adv_frame)
    request_data = i_mgr.build_access_request()

    # 阶段 c: 广播方收到请求, 生成响应
    response_frame, _accepted = b_mgr.handle_access_request(
        request_data, peer_address=initiator_addr,
    )

    # 阶段 d/e: 发起方收到响应
    if response_frame is not None:
        response_bytes = response_frame.pack()
        i_mgr.handle_access_response(response_bytes)

    return b_mgr, i_mgr
