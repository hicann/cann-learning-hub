"""广播帧编解码 -- TXS-10002-2025 标准 7.1.4

实现 SLE 广播、发现和接入所需的帧结构编解码:
- 广播帧通用结构 (BroadcastFrame)
- 扩展广播帧资源配置信息 (ExtAdvResourceConfig)
- 发现接入资源配置信息 (DiscoveryAccessResourceConfig)
- 接入基本信息 (AccessBasicInfo)
- 接入请求信息 (AccessRequestInfo)
- 接入响应信息 (AccessResponseInfo)
- 启动系统管理帧信息 (SystemMgmtFrameInfo)
"""

from __future__ import annotations

__all__ = [
    "AccessBasicInfo",
    "AccessRequestInfo",
    "AccessResponseEntry",
    "AccessResponseInfo",
    "AccessResponseType",
    "AddrType",
    "BroadcastDataType",
    "BroadcastFilter",
    "BroadcastFrame",
    "DiscoveryAccessEntry",
    "DiscoveryAccessResourceConfig",
    "ExtAdvResourceConfig",
    "FilterCondition",
    "FilterOp",
    "GTNegotiation",
    "NarrowbandMeasurementConfig",
    "NonLinkedBroadcastLinkInfo",
    "QueryRequestFilterInfo",
    "RequestType",
    "SystemMgmtFrameInfo",
    "TransportIndicationInfo",
    "UWBPulseMeasurementConfig",
]


import struct
from dataclasses import dataclass, field
from enum import IntEnum

# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------


class AddrType(IntEnum):
    """媒体接入层标识类型 (7.1.4 表25)"""
    ALLIANCE_ASSIGNED = 0
    LOCAL_ORG = 2
    RESOLVABLE_RANDOM = 3
    NON_RESOLVABLE_RANDOM = 4
    ALLIANCE_RESERVED = 5
    PRIVATE = 6


class BroadcastDataType(IntEnum):
    """广播帧数据类型 (7.1.4 表26)"""
    DISCOVERY_ACCESS_RESOURCE = 0x00
    TRANSPORT_INDICATION = 0x01
    ACCESS_BASIC = 0x02
    ACCESS_REQUEST = 0x03
    ACCESS_RESPONSE = 0x04
    SYSTEM_MGMT_FRAME = 0x05
    UNLINKED_BROADCAST_LINK = 0x06
    QUERY_REQUEST_FILTER = 0x07
    NB_HOPPING_MEAS_CONFIG = 0x08
    UWB_PULSE_MEAS_CONFIG = 0x09
    UPPER_LAYER_DATA = 0xFF


class RequestType(IntEnum):
    """请求类型标志 (7.1.4.2)"""
    QUERY = 0
    ACCESS = 1
    UNRESTRICTED = 2


class GTNegotiation(IntEnum):
    """GT 角色协商结果 (7.1.4.2)"""
    NEGOTIATE_T = 0b00
    NEGOTIATE_G = 0b01
    FIXED_T = 0b10
    FIXED_G = 0b11


class AccessResponseType(IntEnum):
    """接入响应类型 (7.1.4.6)"""
    ACCEPT = 0
    GT_NEGOTIATION_FAIL = 1
    RESOURCE_LIMITED = 2
    USER_REJECT = 3


# ---------------------------------------------------------------------------
# 7.1.4.1 扩展广播帧资源配置信息 (4 字节)
# ---------------------------------------------------------------------------


@dataclass
class ExtAdvResourceConfig:
    """扩展广播帧资源配置信息 (7.1.4.1, 表27)

    4 字节结构:
    - 频点索引:       8 bits
    - 偏移量:        12 bits
    - 无线帧类型指示:  4 bits
    - 带宽指示:       2 bits
    - 导频密度指示:    2 bits
    - 时钟精度:       3 bits
    - 偏移量单位:     1 bit
    """
    channel_index: int        # 频点索引 (0-255)
    offset: int               # 偏移量 (0-4095)
    frame_type: int            # 无线帧类型 (0-15)
    bandwidth: int             # 带宽 (0=1M, 1=2M, 2=4M)
    pilot_density: int         # 导频密度 (0=4:1, 1=8:1, 2=16:1, 3=无)
    clock_accuracy: int        # 时钟精度 (0-7)
    offset_unit: int           # 偏移量单位 (0=1基础时隙, 1=10基础时隙)

    BYTE_LENGTH = 4

    def pack(self) -> bytes:
        b0 = self.channel_index & 0xFF
        b1 = (self.offset >> 4) & 0xFF
        b2 = ((self.offset & 0x0F) << 4) | (self.frame_type & 0x0F)
        b3 = (
            ((self.bandwidth & 0x03) << 6)
            | ((self.pilot_density & 0x03) << 4)
            | ((self.clock_accuracy & 0x07) << 1)
            | (self.offset_unit & 0x01)
        )
        return bytes([b0, b1, b2, b3])

    @classmethod
    def unpack(cls, data: bytes) -> ExtAdvResourceConfig:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        channel_index = data[0]
        offset = (data[1] << 4) | ((data[2] >> 4) & 0x0F)
        frame_type = data[2] & 0x0F
        bandwidth = (data[3] >> 6) & 0x03
        pilot_density = (data[3] >> 4) & 0x03
        clock_accuracy = (data[3] >> 1) & 0x07
        offset_unit = data[3] & 0x01
        return cls(
            channel_index, offset, frame_type,
            bandwidth, pilot_density, clock_accuracy, offset_unit,
        )


# ---------------------------------------------------------------------------
# 7.1.4.2 发现接入资源配置信息
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryAccessEntry:
    """单个发现接入资源条目"""
    request_type: int         # 请求类型标志 (2 bits)
    carry_info_indication: int  # 请求携带信息指示 (2 bits)
    peer_addr_type: int       # 对端标识类型 (3 bits)
    addr_present: int         # 标识指示 (1 bit)
    peer_addr: bytes          # 对端标识 (6字节, 仅当 addr_present=1 时有效)


@dataclass
class DiscoveryAccessResourceConfig:
    """发现接入资源配置信息 (7.1.4.2, 表28)"""
    request_offset: int            # 请求偏移量 (2字节, us)
    request_max_length: int        # 请求最大长度 (1字节)
    response_offset: int           # 请求响应偏移量 (2字节, us)
    gt_negotiation: int            # GT 协商指示 (2 bits)
    entry_count: int               # 请求/响应个数 (6 bits)
    entries: list[DiscoveryAccessEntry] = field(default_factory=list)

    def pack(self) -> bytes:
        buf = struct.pack(">HBH", self.request_offset,
                          self.request_max_length, self.response_offset)
        buf += bytes([
            ((self.gt_negotiation & 0x03) << 6) | (self.entry_count & 0x3F)
        ])
        for entry in self.entries:
            flag = (
                ((entry.request_type & 0x03) << 6)
                | ((entry.carry_info_indication & 0x03) << 4)
                | ((entry.peer_addr_type & 0x07) << 1)
                | (entry.addr_present & 0x01)
            )
            buf += bytes([flag])
            if entry.addr_present:
                buf += entry.peer_addr[:6].ljust(6, b"\x00")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> DiscoveryAccessResourceConfig:
        if len(data) < 6:
            raise ValueError("数据不足: 至少需要 6 字节")
        req_offset, req_max_len, resp_offset = struct.unpack(">HBH", data[:5])
        gt_neg = (data[5] >> 6) & 0x03
        count = data[5] & 0x3F
        entries: list[DiscoveryAccessEntry] = []
        pos = 6
        for _ in range(count):
            if pos >= len(data):
                break
            flag = data[pos]
            pos += 1
            req_type = (flag >> 6) & 0x03
            carry_info = (flag >> 4) & 0x03
            peer_type = (flag >> 1) & 0x07
            addr_present = flag & 0x01
            peer_addr = b""
            if addr_present and pos + 6 <= len(data):
                peer_addr = bytes(data[pos:pos + 6])
                pos += 6
            entries.append(DiscoveryAccessEntry(
                req_type, carry_info, peer_type, addr_present, peer_addr,
            ))
        return cls(req_offset, req_max_len, resp_offset, gt_neg, count, entries)


# ---------------------------------------------------------------------------
# 7.1.4.4 接入基本信息 (可变长)
# ---------------------------------------------------------------------------


@dataclass
class AccessBasicInfo:
    """接入基本信息 (7.1.4.4, 表31)"""
    smf_baseline_slot: int         # 系统管理帧基线时隙 (4字节)
    smf_offset: int                # 启动系统管理帧偏移量 (2字节, us)
    smf_link_id: int               # 系统管理帧逻辑链路标识 (4字节)
    smf_period: int                # 系统管理帧周期间隔 (3字节, 基础时隙)
    smf_frame_type: int            # 系统管理帧无线帧类型 (4 bits)
    smf_bandwidth: int             # 系统管理帧带宽 (2 bits)
    smf_pilot_density: int         # 系统管理帧导频密度 (2 bits)
    access_link_id: int            # 接入逻辑链路标识 (3字节)
    access_period: int             # 接入周期间隔 (2字节, 基础时隙)
    access_timeout: int            # 接入超时时间 (1字节)
    sleep_clock_accuracy: int      # 睡眠时钟精度 (3 bits)
    access_crc_type: int           # 接入CRC类型 (1 bit, 0=CRC24, 1=CRC32)
    access_crc_init: int           # 接入CRC初始值 (4字节)
    hop_map: bytes                 # 数据传输可用跳频地图 (10或25字节)
    smf_channel_count: int         # 系统管理帧可用频点个数 (1字节)
    smf_channel_table: bytes       # 系统管理帧频点表 (可变长)

    def pack(self) -> bytes:
        buf = self.smf_baseline_slot.to_bytes(4, "big")
        buf += self.smf_offset.to_bytes(2, "big")
        buf += self.smf_link_id.to_bytes(4, "big")
        buf += self.smf_period.to_bytes(3, "big")
        b_config = (
            ((self.smf_frame_type & 0x0F) << 4)
            | ((self.smf_bandwidth & 0x03) << 2)
            | (self.smf_pilot_density & 0x03)
        )
        buf += bytes([b_config])
        buf += self.access_link_id.to_bytes(3, "big")
        buf += self.access_period.to_bytes(2, "big")
        buf += bytes([self.access_timeout & 0xFF])
        b_flags = (
            ((self.sleep_clock_accuracy & 0x07) << 5)
            | (self.access_crc_type & 0x01)
        )
        buf += bytes([b_flags])
        buf += self.access_crc_init.to_bytes(4, "big")
        buf += self.hop_map
        buf += bytes([self.smf_channel_count & 0xFF])
        buf += self.smf_channel_table
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AccessBasicInfo:
        if len(data) < 25:
            raise ValueError("数据不足: 至少需要 25 字节")
        smf_baseline = int.from_bytes(data[0:4], "big")
        smf_offset = int.from_bytes(data[4:6], "big")
        smf_link_id = int.from_bytes(data[6:10], "big")
        smf_period = int.from_bytes(data[10:13], "big")
        config = data[13]
        smf_ft = (config >> 4) & 0x0F
        smf_bw = (config >> 2) & 0x03
        smf_pd = config & 0x03
        access_lid = int.from_bytes(data[14:17], "big")
        access_period = int.from_bytes(data[17:19], "big")
        access_timeout = data[19]
        flags = data[20]
        sca = (flags >> 5) & 0x07
        crc_type = flags & 0x01
        crc_init = int.from_bytes(data[21:25], "big")
        # 根据跳频地图长度判断: 2.4GHz = 10字节, 5.xGHz = 25字节
        remaining = len(data) - 25
        if remaining >= 25 + 1:
            hop_map = bytes(data[25:50])
            pos = 50
        else:
            hop_map = bytes(data[25:35])
            pos = 35
        smf_ch_count = data[pos] if pos < len(data) else 0
        pos += 1
        smf_ch_table = bytes(data[pos:pos + smf_ch_count])
        return cls(
            smf_baseline, smf_offset, smf_link_id, smf_period,
            smf_ft, smf_bw, smf_pd, access_lid, access_period,
            access_timeout, sca, crc_type, crc_init, hop_map,
            smf_ch_count, smf_ch_table,
        )


# ---------------------------------------------------------------------------
# 7.1.4.5 接入请求信息 (可变长)
# ---------------------------------------------------------------------------


@dataclass
class AccessRequestInfo:
    """接入请求信息 (7.1.4.5, 表32/33)"""
    structure_indication: int   # 结构指示字段 (8 bits 位图)
    gt_role: int | None = None         # GT角色标志 (2 bits)
    frame_support: int | None = None   # 无线帧结构指示 (4 bits 位图)
    bandwidth_support: int | None = None  # 带宽指示 (3 bits 位图)
    mcs_support: int | None = None     # MCS指示 (13 bits 位图)
    pilot_support: int | None = None   # 导频指示 (4 bits 位图)
    slot_support: int | None = None    # 调度时隙指示 (5 bits 位图)
    switch_delay: int | None = None    # 收发切换时延 (4 bits)
    crc_support: int | None = None     # CRC指示 (2 bits 位图)

    def pack(self) -> bytes:
        # 按位收集存在的字段, 组成比特流后按字节对齐
        bits: list[int] = []
        bits.extend(_int_to_bits(self.structure_indication, 8))
        if self.structure_indication & 0x01 and self.gt_role is not None:
            bits.extend(_int_to_bits(self.gt_role, 2))
        if self.structure_indication & 0x02 and self.frame_support is not None:
            bits.extend(_int_to_bits(self.frame_support, 4))
        if self.structure_indication & 0x04 and self.bandwidth_support is not None:
            bits.extend(_int_to_bits(self.bandwidth_support, 3))
        if self.structure_indication & 0x08 and self.mcs_support is not None:
            bits.extend(_int_to_bits(self.mcs_support, 13))
        if self.structure_indication & 0x10 and self.pilot_support is not None:
            bits.extend(_int_to_bits(self.pilot_support, 4))
        if self.structure_indication & 0x20 and self.slot_support is not None:
            bits.extend(_int_to_bits(self.slot_support, 5))
        if self.structure_indication & 0x40 and self.switch_delay is not None:
            bits.extend(_int_to_bits(self.switch_delay, 4))
        if self.structure_indication & 0x80 and self.crc_support is not None:
            bits.extend(_int_to_bits(self.crc_support, 2))
        # 高位补零对齐到字节
        pad = (8 - len(bits) % 8) % 8
        bits.extend([0] * pad)
        return _bits_to_bytes(bits)

    @classmethod
    def unpack(cls, data: bytes) -> AccessRequestInfo:
        if len(data) < 1:
            raise ValueError("数据不足: 至少需要 1 字节")
        bits = _bytes_to_bits(data)
        pos = 0
        si = _bits_to_int(bits, pos, 8)
        pos += 8
        gt = frame = bw = mcs = pilot = slot = switch = crc = None
        if si & 0x01:
            gt = _bits_to_int(bits, pos, 2)
            pos += 2
        if si & 0x02:
            frame = _bits_to_int(bits, pos, 4)
            pos += 4
        if si & 0x04:
            bw = _bits_to_int(bits, pos, 3)
            pos += 3
        if si & 0x08:
            mcs = _bits_to_int(bits, pos, 13)
            pos += 13
        if si & 0x10:
            pilot = _bits_to_int(bits, pos, 4)
            pos += 4
        if si & 0x20:
            slot = _bits_to_int(bits, pos, 5)
            pos += 5
        if si & 0x40:
            switch = _bits_to_int(bits, pos, 4)
            pos += 4
        if si & 0x80:
            crc = _bits_to_int(bits, pos, 2)
            pos += 2
        return cls(si, gt, frame, bw, mcs, pilot, slot, switch, crc)


# ---------------------------------------------------------------------------
# 7.1.4.6 接入响应信息 (可变长)
# ---------------------------------------------------------------------------


@dataclass
class AccessResponseEntry:
    """单个接入响应条目"""
    peer_addr: bytes               # 对端标识 (6字节)
    response_type: int             # 请求响应类型 (3 bits)
    peer_addr_type: int            # 对端标识类型 (3 bits)
    repeat_indication: int         # 重复指示 (2 bits)
    capability: bytes | None = None  # 广播节点能力指示 (6字节, 可选)


@dataclass
class AccessResponseInfo:
    """接入响应信息 (7.1.4.6, 表34)"""
    entries: list[AccessResponseEntry] = field(default_factory=list)

    def pack(self) -> bytes:
        count = len(self.entries)
        buf = bytes([(count & 0x3F) << 2])
        for entry in self.entries:
            buf += entry.peer_addr[:6].ljust(6, b"\x00")
            flag = (
                ((entry.response_type & 0x07) << 5)
                | ((entry.peer_addr_type & 0x07) << 2)
                | (entry.repeat_indication & 0x03)
            )
            buf += bytes([flag])
            if entry.repeat_indication in (1, 2) and entry.capability:
                buf += entry.capability[:6].ljust(6, b"\x00")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AccessResponseInfo:
        if len(data) < 1:
            raise ValueError("数据不足")
        count = (data[0] >> 2) & 0x3F
        entries: list[AccessResponseEntry] = []
        pos = 1
        for _ in range(count):
            if pos + 7 > len(data):
                break
            peer_addr = bytes(data[pos:pos + 6])
            pos += 6
            flag = data[pos]
            pos += 1
            resp_type = (flag >> 5) & 0x07
            peer_type = (flag >> 2) & 0x07
            repeat = flag & 0x03
            cap = None
            if repeat in (1, 2) and pos + 6 <= len(data):
                cap = bytes(data[pos:pos + 6])
                pos += 6
            entries.append(AccessResponseEntry(
                peer_addr, resp_type, peer_type, repeat, cap,
            ))
        return cls(entries)


# ---------------------------------------------------------------------------
# 7.1.4.7 启动系统管理帧信息
# ---------------------------------------------------------------------------


@dataclass
class SystemMgmtFrameInfo:
    """启动系统管理帧信息 (7.1.4.7, 表35)"""
    baseline_slot: int           # 系统管理帧基线时隙 (4字节)
    offset: int                  # 偏移量 (2字节, us)
    access_addr: int             # 系统管理帧接入地址 (3字节)
    period: int                  # 周期间隔 (3字节, 基础时隙)
    frame_type: int              # 无线帧类型 (4 bits)
    bandwidth: int               # 带宽 (2 bits)
    pilot_density: int           # 导频密度 (2 bits)
    channel_count: int           # 可用频点个数 (1字节)
    channel_table: bytes         # 频点表 (可变长)

    def pack(self) -> bytes:
        buf = self.baseline_slot.to_bytes(4, "big")
        buf += self.offset.to_bytes(2, "big")
        buf += self.access_addr.to_bytes(3, "big")
        buf += self.period.to_bytes(3, "big")
        config = (
            ((self.frame_type & 0x0F) << 4)
            | ((self.bandwidth & 0x03) << 2)
            | (self.pilot_density & 0x03)
        )
        buf += bytes([config])
        buf += bytes([self.channel_count & 0xFF])
        buf += self.channel_table
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> SystemMgmtFrameInfo:
        if len(data) < 14:
            raise ValueError("数据不足: 至少需要 14 字节")
        baseline = int.from_bytes(data[0:4], "big")
        offset = int.from_bytes(data[4:6], "big")
        access_addr = int.from_bytes(data[6:9], "big")
        period = int.from_bytes(data[9:12], "big")
        config = data[12]
        ft = (config >> 4) & 0x0F
        bw = (config >> 2) & 0x03
        pd = config & 0x03
        ch_count = data[13]
        ch_table = bytes(data[14:14 + ch_count])
        return cls(baseline, offset, access_addr, period, ft, bw, pd,
                   ch_count, ch_table)


# ---------------------------------------------------------------------------
# 7.1.4.3 传输指示信息
# ---------------------------------------------------------------------------


@dataclass
class TransportIndicationInfo:
    """传输指示信息 (7.1.4.3, 表29)

    总长度 480 bits (2.4GHz) 或 600 bits (5.xGHz),
    取决于跳频地图长度 (80 或 200 bits)。
    """
    system_slot_seq: int           # 系统基础时隙顺序号 (32 bits)
    event_group_offset: int        # 事件组起始偏移时间 (24 bits, us)
    event_group_period: int        # 事件组周期 (16 bits, 调度时隙)
    event_period: int              # 事件周期 (16 bits, 调度时隙)
    intra_event_interval: int      # 事件内间隔 (16 bits, us)
    inter_event_interval: int      # 事件间间隔和事件组间间隔 (16 bits, us)
    event_count: int               # 事件总数 (8 bits)
    peer_addr: bytes               # 对端媒体接入层标识 (48 bits / 6字节)
    peer_addr_type: int            # 对端标识类型 (3 bits)
    sleep_clock_accuracy: int      # 睡眠时钟精度 (3 bits)
    first_last_indication: int     # 先发后发指示 (1 bit)
    reserved_1: int = 0            # 保留 (1 bit)
    tx_frame_type: int = 0         # 先发链路无线帧类型指示 (4 bits)
    rx_frame_type: int = 0         # 后发链路无线帧类型指示 (4 bits)
    tx_crc_type: int = 0           # 先发链路CRC类型 (1 bit)
    rx_crc_type: int = 0           # 后发链路CRC类型 (1 bit)
    tx_feedback_type: int = 0      # 先发链路反馈类型指示 (6 bits)
    rx_feedback_type: int = 0      # 后发链路反馈类型指示 (3 bits)
    system_schedule_slot: int = 0  # 系统调度时隙 (3 bits)
    reserved_2: int = 0            # 保留 (2 bits)
    tx_link_id: int = 0            # 先发链路逻辑链路标识 (24 bits)
    rx_link_id: int = 0            # 后发链路逻辑链路标识 (24 bits)
    tx_bandwidth: int = 0          # 先发链路带宽指示 (2 bits)
    rx_bandwidth: int = 0          # 后发链路带宽指示 (2 bits)
    tx_pilot_density: int = 0      # 先发链路导频密度指示 (2 bits)
    rx_pilot_density: int = 0      # 后发链路导频密度指示 (2 bits)
    tx_pdu_max: int = 0            # 先发链路PDU最大值 (11 bits)
    rx_pdu_max: int = 0            # 后发链路PDU最大值 (11 bits)
    tx_max_time_offset: int = 0    # 先发链路最大时间偏移量 (9 bits)
    rx_max_time_offset: int = 0    # 后发链路最大时间偏移量 (9 bits)
    tx_crc_init: int = 0           # 先发链路CRC初始值 (32 bits)
    rx_crc_init: int = 0           # 后发链路CRC初始值 (32 bits)
    delay_period: int = 0          # 延迟周期 (16 bits)
    timeout: int = 0               # 超时时间 (16 bits)
    hop_map: bytes = b""           # 跳频地图 (10 或 25 字节)
    is_5g: bool = False            # 是否使用 5.xGHz 跳频地图

    def pack(self) -> bytes:
        """编码为字节序列"""
        bits: list[int] = []
        bits.extend(_int_to_bits(self.system_slot_seq, 32))
        bits.extend(_int_to_bits(self.event_group_offset, 24))
        bits.extend(_int_to_bits(self.event_group_period, 16))
        bits.extend(_int_to_bits(self.event_period, 16))
        bits.extend(_int_to_bits(self.intra_event_interval, 16))
        bits.extend(_int_to_bits(self.inter_event_interval, 16))
        bits.extend(_int_to_bits(self.event_count, 8))
        # peer_addr: 48 bits
        for b in self.peer_addr[:6].ljust(6, b"\x00"):
            bits.extend(_int_to_bits(b, 8))
        bits.extend(_int_to_bits(self.peer_addr_type, 3))
        bits.extend(_int_to_bits(self.sleep_clock_accuracy, 3))
        bits.extend(_int_to_bits(self.first_last_indication, 1))
        bits.extend(_int_to_bits(self.reserved_1, 1))
        bits.extend(_int_to_bits(self.tx_frame_type, 4))
        bits.extend(_int_to_bits(self.rx_frame_type, 4))
        bits.extend(_int_to_bits(self.tx_crc_type, 1))
        bits.extend(_int_to_bits(self.rx_crc_type, 1))
        bits.extend(_int_to_bits(self.tx_feedback_type, 6))
        bits.extend(_int_to_bits(self.rx_feedback_type, 3))
        bits.extend(_int_to_bits(self.system_schedule_slot, 3))
        bits.extend(_int_to_bits(self.reserved_2, 2))
        bits.extend(_int_to_bits(self.tx_link_id, 24))
        bits.extend(_int_to_bits(self.rx_link_id, 24))
        bits.extend(_int_to_bits(self.tx_bandwidth, 2))
        bits.extend(_int_to_bits(self.rx_bandwidth, 2))
        bits.extend(_int_to_bits(self.tx_pilot_density, 2))
        bits.extend(_int_to_bits(self.rx_pilot_density, 2))
        bits.extend(_int_to_bits(self.tx_pdu_max, 11))
        bits.extend(_int_to_bits(self.rx_pdu_max, 11))
        bits.extend(_int_to_bits(self.tx_max_time_offset, 9))
        bits.extend(_int_to_bits(self.rx_max_time_offset, 9))
        bits.extend(_int_to_bits(self.tx_crc_init, 32))
        bits.extend(_int_to_bits(self.rx_crc_init, 32))
        bits.extend(_int_to_bits(self.delay_period, 16))
        bits.extend(_int_to_bits(self.timeout, 16))
        # hop_map: 80 or 200 bits
        hop_len = 25 if self.is_5g else 10
        hop = self.hop_map[:hop_len].ljust(hop_len, b"\x00")
        for b in hop:
            bits.extend(_int_to_bits(b, 8))
        return _bits_to_bytes(bits)

    @classmethod
    def unpack(
        cls, data: bytes, is_5g: bool = False,
    ) -> TransportIndicationInfo:
        """从字节序列解码"""
        hop_len = 25 if is_5g else 10
        expected_bits = 400 + hop_len * 8
        if len(data) * 8 < expected_bits:
            raise ValueError(
                f"数据不足: 需要至少 {expected_bits} bits "
                f"({(expected_bits + 7) // 8} 字节)"
            )
        bits = _bytes_to_bits(data)
        p = 0

        def _take(w: int) -> int:
            nonlocal p
            v = _bits_to_int(bits, p, w)
            p += w
            return v

        system_slot_seq = _take(32)
        event_group_offset = _take(24)
        event_group_period = _take(16)
        event_period = _take(16)
        intra_event_interval = _take(16)
        inter_event_interval = _take(16)
        event_count = _take(8)
        peer_addr = bytes([_take(8) for _ in range(6)])
        peer_addr_type = _take(3)
        sleep_clock_accuracy = _take(3)
        first_last_indication = _take(1)
        reserved_1 = _take(1)
        tx_frame_type = _take(4)
        rx_frame_type = _take(4)
        tx_crc_type = _take(1)
        rx_crc_type = _take(1)
        tx_feedback_type = _take(6)
        rx_feedback_type = _take(3)
        system_schedule_slot = _take(3)
        reserved_2 = _take(2)
        tx_link_id = _take(24)
        rx_link_id = _take(24)
        tx_bandwidth = _take(2)
        rx_bandwidth = _take(2)
        tx_pilot_density = _take(2)
        rx_pilot_density = _take(2)
        tx_pdu_max = _take(11)
        rx_pdu_max = _take(11)
        tx_max_time_offset = _take(9)
        rx_max_time_offset = _take(9)
        tx_crc_init = _take(32)
        rx_crc_init = _take(32)
        delay_period = _take(16)
        timeout = _take(16)
        hop_map = bytes([_take(8) for _ in range(hop_len)])
        return cls(
            system_slot_seq=system_slot_seq,
            event_group_offset=event_group_offset,
            event_group_period=event_group_period,
            event_period=event_period,
            intra_event_interval=intra_event_interval,
            inter_event_interval=inter_event_interval,
            event_count=event_count,
            peer_addr=peer_addr,
            peer_addr_type=peer_addr_type,
            sleep_clock_accuracy=sleep_clock_accuracy,
            first_last_indication=first_last_indication,
            reserved_1=reserved_1,
            tx_frame_type=tx_frame_type,
            rx_frame_type=rx_frame_type,
            tx_crc_type=tx_crc_type,
            rx_crc_type=rx_crc_type,
            tx_feedback_type=tx_feedback_type,
            rx_feedback_type=rx_feedback_type,
            system_schedule_slot=system_schedule_slot,
            reserved_2=reserved_2,
            tx_link_id=tx_link_id,
            rx_link_id=rx_link_id,
            tx_bandwidth=tx_bandwidth,
            rx_bandwidth=rx_bandwidth,
            tx_pilot_density=tx_pilot_density,
            rx_pilot_density=rx_pilot_density,
            tx_pdu_max=tx_pdu_max,
            rx_pdu_max=rx_pdu_max,
            tx_max_time_offset=tx_max_time_offset,
            rx_max_time_offset=rx_max_time_offset,
            tx_crc_init=tx_crc_init,
            rx_crc_init=rx_crc_init,
            delay_period=delay_period,
            timeout=timeout,
            hop_map=hop_map,
            is_5g=is_5g,
        )


# ---------------------------------------------------------------------------
# 7.1.4.8 非链接态广播链路信息
# ---------------------------------------------------------------------------


@dataclass
class NonLinkedBroadcastLinkInfo:
    """非链接态广播链路信息 (7.1.4.8, 表36/37)"""
    transmission_type: int         # 传输类型 (1 bit, 0=异步, 1=同步)
    service_adapt_mode: int        # 业务适配方式 (1 bit, 0=周期, 1=非周期)
    reserved: int = 0              # 保留 (6 bits)
    system_slot_seq: int = 0       # 系统基础时隙顺序号 (32 bits)
    event_group_offset: int = 0    # 事件组起始偏移时间 (24 bits, us)
    event_group_set_id: int = 0    # 事件组集合标识 (8 bits)
    event_group_count: int = 0     # 事件组个数 (8 bits)
    event_group_interval: int = 0  # 事件组间隔 (8 bits, 基础时隙)
    event_group_period: int = 0    # 事件组周期 (16 bits, 基础时隙)
    event_period: int = 0          # 事件周期 (16 bits, 基础时隙)
    event_count: int = 0           # 事件总数 (8 bits)
    sync_anchor_delay: int = 0     # 同步锚点延迟 (24 bits, us)
    sync_ref_delay: int = 0        # 同步参考延迟 (24 bits, us)
    base_link_id: int = 0          # 基础逻辑链路标识 (24 bits)
    frame_type: int = 0            # 无线帧类型 (4 bits)
    bandwidth: int = 0             # 带宽指示 (2 bits)
    pilot_density: int = 0         # 导频密度指示 (2 bits)
    sdu_max: int = 0               # SDU最大值 (12 bits)
    sdu_period: int = 0            # SDU周期 (20 bits, us)
    pdu_max: int = 0               # PDU最大值 (11 bits)
    new_packet_count: int = 0      # 新数据包计数 (4 bits)
    crc_type: int = 0              # CRC类型 (1 bit)
    crc_base_init: int = 0         # CRC基础初始值 (32 bits)
    hop_map: bytes = b""           # 跳频地图 (10 或 25 字节)
    giv: bytes = b""               # 加密使用的GIV (8 字节)
    gskd: bytes = b""              # 生成加密密钥的GSKD (16 字节)
    is_5g: bool = False            # 是否使用 5.xGHz 跳频地图

    def pack(self) -> bytes:
        """编码为字节序列"""
        bits: list[int] = []
        bits.extend(_int_to_bits(self.transmission_type, 1))
        bits.extend(_int_to_bits(self.service_adapt_mode, 1))
        bits.extend(_int_to_bits(self.reserved, 6))
        bits.extend(_int_to_bits(self.system_slot_seq, 32))
        bits.extend(_int_to_bits(self.event_group_offset, 24))
        bits.extend(_int_to_bits(self.event_group_set_id, 8))
        bits.extend(_int_to_bits(self.event_group_count, 8))
        bits.extend(_int_to_bits(self.event_group_interval, 8))
        bits.extend(_int_to_bits(self.event_group_period, 16))
        bits.extend(_int_to_bits(self.event_period, 16))
        bits.extend(_int_to_bits(self.event_count, 8))
        bits.extend(_int_to_bits(self.sync_anchor_delay, 24))
        bits.extend(_int_to_bits(self.sync_ref_delay, 24))
        bits.extend(_int_to_bits(self.base_link_id, 24))
        bits.extend(_int_to_bits(self.frame_type, 4))
        bits.extend(_int_to_bits(self.bandwidth, 2))
        bits.extend(_int_to_bits(self.pilot_density, 2))
        bits.extend(_int_to_bits(self.sdu_max, 12))
        bits.extend(_int_to_bits(self.sdu_period, 20))
        bits.extend(_int_to_bits(self.pdu_max, 11))
        bits.extend(_int_to_bits(self.new_packet_count, 4))
        bits.extend(_int_to_bits(self.crc_type, 1))
        bits.extend(_int_to_bits(self.crc_base_init, 32))
        # hop_map
        hop_len = 25 if self.is_5g else 10
        hop = self.hop_map[:hop_len].ljust(hop_len, b"\x00")
        for b in hop:
            bits.extend(_int_to_bits(b, 8))
        # giv: 8 字节
        giv = self.giv[:8].ljust(8, b"\x00")
        for b in giv:
            bits.extend(_int_to_bits(b, 8))
        # gskd: 16 字节
        gskd = self.gskd[:16].ljust(16, b"\x00")
        for b in gskd:
            bits.extend(_int_to_bits(b, 8))
        return _bits_to_bytes(bits)

    @classmethod
    def unpack(
        cls, data: bytes, is_5g: bool = False,
    ) -> NonLinkedBroadcastLinkInfo:
        """从字节序列解码"""
        hop_len = 25 if is_5g else 10
        # 固定位: 288 + hop + giv(64) + gskd(128)
        fixed_bits = 288 + hop_len * 8 + 64 + 128
        if len(data) * 8 < fixed_bits:
            raise ValueError(
                f"数据不足: 需要至少 {fixed_bits} bits "
                f"({(fixed_bits + 7) // 8} 字节)"
            )
        bits = _bytes_to_bits(data)
        p = 0

        def _take(w: int) -> int:
            nonlocal p
            v = _bits_to_int(bits, p, w)
            p += w
            return v

        transmission_type = _take(1)
        service_adapt_mode = _take(1)
        reserved = _take(6)
        system_slot_seq = _take(32)
        event_group_offset = _take(24)
        event_group_set_id = _take(8)
        event_group_count = _take(8)
        event_group_interval = _take(8)
        event_group_period = _take(16)
        event_period = _take(16)
        event_count = _take(8)
        sync_anchor_delay = _take(24)
        sync_ref_delay = _take(24)
        base_link_id = _take(24)
        frame_type = _take(4)
        bandwidth = _take(2)
        pilot_density = _take(2)
        sdu_max = _take(12)
        sdu_period = _take(20)
        pdu_max = _take(11)
        new_packet_count = _take(4)
        crc_type = _take(1)
        crc_base_init = _take(32)
        hop_map = bytes([_take(8) for _ in range(hop_len)])
        giv = bytes([_take(8) for _ in range(8)])
        gskd = bytes([_take(8) for _ in range(16)])
        return cls(
            transmission_type=transmission_type,
            service_adapt_mode=service_adapt_mode,
            reserved=reserved,
            system_slot_seq=system_slot_seq,
            event_group_offset=event_group_offset,
            event_group_set_id=event_group_set_id,
            event_group_count=event_group_count,
            event_group_interval=event_group_interval,
            event_group_period=event_group_period,
            event_period=event_period,
            event_count=event_count,
            sync_anchor_delay=sync_anchor_delay,
            sync_ref_delay=sync_ref_delay,
            base_link_id=base_link_id,
            frame_type=frame_type,
            bandwidth=bandwidth,
            pilot_density=pilot_density,
            sdu_max=sdu_max,
            sdu_period=sdu_period,
            pdu_max=pdu_max,
            new_packet_count=new_packet_count,
            crc_type=crc_type,
            crc_base_init=crc_base_init,
            hop_map=hop_map,
            giv=giv,
            gskd=gskd,
            is_5g=is_5g,
        )


# ---------------------------------------------------------------------------
# 7.1.4.9 查询请求过滤信息
# ---------------------------------------------------------------------------


@dataclass
class QueryRequestFilterInfo:
    """查询请求过滤信息 (7.1.4.9, 表38)

    变长结构, 包含:
    - 16比特标准服务标识列表
    - 128比特自定义服务标识列表
    """
    uuid_16_list: list[int] = field(default_factory=list)
    uuid_128_list: list[bytes] = field(default_factory=list)

    def pack(self) -> bytes:
        """编码为字节序列"""
        buf = b""
        # 16-bit UUID 个数 (1字节)
        count_16 = len(self.uuid_16_list)
        buf += bytes([count_16 & 0xFF])
        for uuid_16 in self.uuid_16_list:
            buf += struct.pack(">H", uuid_16 & 0xFFFF)
        # 128-bit UUID 个数 (1字节)
        count_128 = len(self.uuid_128_list)
        buf += bytes([count_128 & 0xFF])
        for uuid_128 in self.uuid_128_list:
            buf += uuid_128[:16].ljust(16, b"\x00")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> QueryRequestFilterInfo:
        """从字节序列解码"""
        if len(data) < 1:
            raise ValueError("数据不足: 至少需要 1 字节")
        pos = 0
        count_16 = data[pos]
        pos += 1
        uuid_16_list: list[int] = []
        for _ in range(count_16):
            if pos + 2 > len(data):
                break
            uuid_16 = struct.unpack(">H", data[pos:pos + 2])[0]
            uuid_16_list.append(uuid_16)
            pos += 2
        uuid_128_list: list[bytes] = []
        if pos < len(data):
            count_128 = data[pos]
            pos += 1
            for _ in range(count_128):
                if pos + 16 > len(data):
                    break
                uuid_128_list.append(bytes(data[pos:pos + 16]))
                pos += 16
        return cls(uuid_16_list, uuid_128_list)


# ---------------------------------------------------------------------------
# 7.1.4 广播帧通用结构
# ---------------------------------------------------------------------------


@dataclass
class BroadcastFrame:
    """广播帧通用结构 (7.1.4, 表25)

    结构指示 (5 bits) 控制可选字段:
    - bit 0: 本端标识存在
    - bit 1: 解析密钥标识存在
    - bit 2: 对端标识存在
    - bit 3: 扩展广播帧资源配置存在
    - bit 4: 数据部分存在
    """
    structure_indication: int       # 广播帧结构指示 (5 bits)
    local_addr_type: int            # 本端标识类型 (3 bits)
    peer_addr_type: int             # 对端标识类型 (3 bits)
    local_addr: bytes               # 本端标识 (6字节)
    irk_id: int                     # 解析密钥标识 (1字节)
    peer_addr: bytes                # 对端标识 (6字节)
    ext_adv_config: ExtAdvResourceConfig | None = None
    data_items: list[tuple[int, bytes]] = field(default_factory=list)

    def pack(self) -> bytes:
        si = self.structure_indication & 0x1F
        b0 = (si << 3)
        b1 = (
            ((self.local_addr_type & 0x07) << 5)
            | ((self.peer_addr_type & 0x07) << 2)
        )
        buf = bytes([b0, b1])
        if si & 0x01:
            buf += self.local_addr[:6].ljust(6, b"\x00")
        if si & 0x02:
            buf += bytes([self.irk_id & 0xFF])
        if si & 0x04:
            buf += self.peer_addr[:6].ljust(6, b"\x00")
        if si & 0x08 and self.ext_adv_config:
            buf += self.ext_adv_config.pack()
        if si & 0x10:
            for dtype, content in self.data_items:
                buf += bytes([dtype & 0xFF, len(content) & 0xFF])
                buf += content
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> BroadcastFrame:
        if len(data) < 2:
            raise ValueError("数据不足: 至少需要 2 字节")
        si = (data[0] >> 3) & 0x1F
        local_type = (data[1] >> 5) & 0x07
        peer_type = (data[1] >> 2) & 0x07
        pos = 2
        local_addr = b"\x00" * 6
        irk_id = 0
        peer_addr = b"\x00" * 6
        ext_config = None
        items: list[tuple[int, bytes]] = []

        if si & 0x01:
            local_addr = bytes(data[pos:pos + 6])
            pos += 6
        if si & 0x02:
            irk_id = data[pos]
            pos += 1
        if si & 0x04:
            peer_addr = bytes(data[pos:pos + 6])
            pos += 6
        if si & 0x08:
            ext_config = ExtAdvResourceConfig.unpack(data[pos:pos + 4])
            pos += 4
        if si & 0x10:
            while pos + 2 <= len(data):
                dtype = data[pos]
                dlen = data[pos + 1]
                pos += 2
                if pos + dlen > len(data):
                    break
                content = bytes(data[pos:pos + dlen])
                pos += dlen
                items.append((dtype, content))

        return cls(si, local_type, peer_type, local_addr,
                   irk_id, peer_addr, ext_config, items)


# ---------------------------------------------------------------------------
# 7.1.5 广播帧过滤
# ---------------------------------------------------------------------------


class FilterOp(IntEnum):
    """过滤条件组合运算符"""
    AND = 0
    OR = 1
    NOT = 2


@dataclass
class FilterCondition:
    """单个过滤条件"""
    field_name: str          # 匹配字段: address, name, uuid, service_data, vendor_data
    value: bytes = b""       # 匹配值
    negate: bool = False     # 是否取反


@dataclass
class BroadcastFilter:
    """广播帧过滤器 (7.1.5)

    支持按设备地址、设备名称、服务UUID、服务数据和厂商数据等信息
    的单个条件或多个条件的与/或/非组合过滤。
    """
    conditions: list[FilterCondition] = field(default_factory=list)
    operator: FilterOp = FilterOp.AND

    def match(self, frame: BroadcastFrame) -> bool:
        """检查广播帧是否满足过滤条件"""
        if not self.conditions:
            return True

        results = [self._eval_condition(c, frame) for c in self.conditions]

        if self.operator == FilterOp.AND:
            return all(results)
        elif self.operator == FilterOp.OR:
            return any(results)
        else:  # NOT: 对第一个条件取反
            return not results[0] if results else True

    @staticmethod
    def _eval_condition(cond: FilterCondition, frame: BroadcastFrame) -> bool:
        matched = False
        if cond.field_name == "address":
            matched = frame.local_addr == cond.value
        elif cond.field_name == "name":
            for _dtype, content in frame.data_items:
                if content == cond.value:
                    matched = True
                    break
        elif cond.field_name in ("uuid", "service_data", "vendor_data"):
            for _dtype, content in frame.data_items:
                if cond.value in content:
                    matched = True
                    break
        return not matched if cond.negate else matched


# ---------------------------------------------------------------------------
# 7.1.4.10 非链接态窄带跳频测量信息配置
# ---------------------------------------------------------------------------


@dataclass
class NarrowbandMeasurementConfig:
    """非链接态窄带跳频测量信息配置 (7.1.4.10, 表39)

    包含感知信号配置、事件组参数、跳频配置、初始化阶段参数、
    先发链路参数和跳频信道位图等字段。
    """
    config_index: int = 0                     # 感知信号配置索引 (8b)
    event_group_start_offset: int = 0         # 事件组起始偏移时间 (24b, us)
    nb_event_period: int = 0                  # 窄带跳频感知事件周期 (16b)
    nb_event_group_period: int = 0            # 窄带跳频感知事件组周期 (16b)
    nb_events_in_group: int = 0               # 事件组内事件总数 (8b)
    nb_event_group_count: int = 0             # 事件组总数 (8b)
    nb_event_stat_count: int = 0              # 事件统计数量 (8b)
    schedule_slot: int = 4                    # 系统调度时隙 (3b, 默认125us)
    meas_signal_bw: int = 0                   # 测量信号带宽指示 (2b)
    hopping_mode: int = 0                     # 跳频方式指示 (2b)
    init_channel: int = 0                     # 初始化阶段的频点 (10b)
    init_interaction_type: int = 0            # 初始化阶段交互类型指示 (2b)
    init_sync_signal: int = 0                 # 初始化阶段同步信号指示 (4b)
    init_sync_config: int = 0                 # 初始化阶段同步信号配置 (24b)
    init_meas_signal_type: int = 0            # 初始化阶段测量信号类型指示 (1b)
    init_meas_signal1_len: int = 0            # 初始化阶段的测量信号1的长度 (3b)
    init_intra_event_interval: int = 0        # 初始化阶段事件内间隔 (16b, us)
    init_switch_interval: int = 0             # 初始化阶段的测量帧内部切换间隔 (8b, us)
    init_meas_signal2_len: int = 0            # 初始化阶段的测量信号2的长度 (8b)
    init_inter_event_interval: int = 0        # 初始化阶段事件间间隔 (16b, us)
    mf1_inter_event_interval: int = 0         # 测量帧类型1事件间间隔 (16b, us)
    mf2_inter_event_interval: int = 0         # 测量帧类型2事件间间隔 (16b, us)
    nb_intra_event_interval: int = 0          # 窄带跳频感知事件内间隔 (16b, us)
    nb_inter_group_interval: int = 0          # 窄带跳频感知事件组间间隔 (24b, us)
    nb_event_total: int = 0                   # 窄带跳频感知事件总数 (8b)
    mf1_event_period: int = 0                 # 测量帧类型1事件的周期 (8b)
    mf1_switch_interval: int = 0              # 测量帧类型1内部切换间隔 (8b, us)
    tx_sync_signal: int = 0                   # 先发链路同步信号指示 (4b)
    tx_antenna_count: int = 0                 # 先发节点发送天线数量指示 (4b)
    tx_sync_config: int = 0                   # 先发链路同步信号配置 (24b)
    tx_meas_signal_type: int = 0              # 先发链路测量信号类型指示 (1b)
    tx_meas_signal1_len: int = 0              # 先发链路测量信号1长度指示 (3b)
    tx_meas_signal1_security: int = 0         # 先发链路测量信号1安全类型指示 (2b)
    tx_meas_signal2_multitone: int = 0        # 先发链路测量信号2多音指示 (2b)
    tx_first_sub_signal_len: int = 0          # 先发链路第一个子测量信号的长度指示 (8b)
    tx_antenna_switch_interval: int = 0       # 先发节点天线切换间隔 (8b, us)
    tx_meas_signal2_len: int = 0              # 先发链路测量信号2长度指示 (8b)
    hop_band_bitmap: int = 0                  # 跳频信道频带指示 (8b, 位图)
    hop_channel_2g4: int = 0                  # 2.4GHz跳频信道指示 (80b)
    hop_channel_5g1: int = 0                  # 5.1GHz跳频信道指示 (200b)
    hop_channel_5g8: int = 0                  # 5.8GHz跳频信道指示 (128b)
    stability: int = 0                        # 稳定性指示 (1b)
    reserved: int = 0                         # 预留 (4b)

    def pack(self) -> bytes:
        bits: list[int] = []
        bits.extend(_int_to_bits(self.config_index, 8))
        bits.extend(_int_to_bits(self.event_group_start_offset, 24))
        bits.extend(_int_to_bits(self.nb_event_period, 16))
        bits.extend(_int_to_bits(self.nb_event_group_period, 16))
        bits.extend(_int_to_bits(self.nb_events_in_group, 8))
        bits.extend(_int_to_bits(self.nb_event_group_count, 8))
        bits.extend(_int_to_bits(self.nb_event_stat_count, 8))
        bits.extend(_int_to_bits(self.schedule_slot, 3))
        bits.extend(_int_to_bits(self.meas_signal_bw, 2))
        bits.extend(_int_to_bits(self.hopping_mode, 2))
        bits.extend(_int_to_bits(self.init_channel, 10))
        bits.extend(_int_to_bits(self.init_interaction_type, 2))
        bits.extend(_int_to_bits(self.init_sync_signal, 4))
        bits.extend(_int_to_bits(self.init_sync_config, 24))
        bits.extend(_int_to_bits(self.init_meas_signal_type, 1))
        bits.extend(_int_to_bits(self.init_meas_signal1_len, 3))
        bits.extend(_int_to_bits(self.init_intra_event_interval, 16))
        bits.extend(_int_to_bits(self.init_switch_interval, 8))
        bits.extend(_int_to_bits(self.init_meas_signal2_len, 8))
        bits.extend(_int_to_bits(self.init_inter_event_interval, 16))
        bits.extend(_int_to_bits(self.mf1_inter_event_interval, 16))
        bits.extend(_int_to_bits(self.mf2_inter_event_interval, 16))
        bits.extend(_int_to_bits(self.nb_intra_event_interval, 16))
        bits.extend(_int_to_bits(self.nb_inter_group_interval, 24))
        bits.extend(_int_to_bits(self.nb_event_total, 8))
        bits.extend(_int_to_bits(self.mf1_event_period, 8))
        bits.extend(_int_to_bits(self.mf1_switch_interval, 8))
        bits.extend(_int_to_bits(self.tx_sync_signal, 4))
        bits.extend(_int_to_bits(self.tx_antenna_count, 4))
        bits.extend(_int_to_bits(self.tx_sync_config, 24))
        bits.extend(_int_to_bits(self.tx_meas_signal_type, 1))
        bits.extend(_int_to_bits(self.tx_meas_signal1_len, 3))
        bits.extend(_int_to_bits(self.tx_meas_signal1_security, 2))
        bits.extend(_int_to_bits(self.tx_meas_signal2_multitone, 2))
        bits.extend(_int_to_bits(self.tx_first_sub_signal_len, 8))
        bits.extend(_int_to_bits(self.tx_antenna_switch_interval, 8))
        bits.extend(_int_to_bits(self.tx_meas_signal2_len, 8))
        bits.extend(_int_to_bits(self.hop_band_bitmap, 8))
        # 根据频带位图决定包含哪些信道指示
        if self.hop_band_bitmap & 0x01:
            bits.extend(_int_to_bits(self.hop_channel_2g4, 80))
        if self.hop_band_bitmap & 0x02:
            bits.extend(_int_to_bits(self.hop_channel_5g1, 200))
        if self.hop_band_bitmap & 0x04:
            bits.extend(_int_to_bits(self.hop_channel_5g8, 128))
        bits.extend(_int_to_bits(self.stability, 1))
        bits.extend(_int_to_bits(self.reserved, 4))
        # 补齐到字节边界
        pad = (8 - len(bits) % 8) % 8
        bits.extend([0] * pad)
        return _bits_to_bytes(bits)

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasurementConfig:
        bits = _bytes_to_bits(data)
        p = 0

        def _take(w: int) -> int:
            nonlocal p
            v = _bits_to_int(bits, p, w)
            p += w
            return v

        obj = cls()
        obj.config_index = _take(8)
        obj.event_group_start_offset = _take(24)
        obj.nb_event_period = _take(16)
        obj.nb_event_group_period = _take(16)
        obj.nb_events_in_group = _take(8)
        obj.nb_event_group_count = _take(8)
        obj.nb_event_stat_count = _take(8)
        obj.schedule_slot = _take(3)
        obj.meas_signal_bw = _take(2)
        obj.hopping_mode = _take(2)
        obj.init_channel = _take(10)
        obj.init_interaction_type = _take(2)
        obj.init_sync_signal = _take(4)
        obj.init_sync_config = _take(24)
        obj.init_meas_signal_type = _take(1)
        obj.init_meas_signal1_len = _take(3)
        obj.init_intra_event_interval = _take(16)
        obj.init_switch_interval = _take(8)
        obj.init_meas_signal2_len = _take(8)
        obj.init_inter_event_interval = _take(16)
        obj.mf1_inter_event_interval = _take(16)
        obj.mf2_inter_event_interval = _take(16)
        obj.nb_intra_event_interval = _take(16)
        obj.nb_inter_group_interval = _take(24)
        obj.nb_event_total = _take(8)
        obj.mf1_event_period = _take(8)
        obj.mf1_switch_interval = _take(8)
        obj.tx_sync_signal = _take(4)
        obj.tx_antenna_count = _take(4)
        obj.tx_sync_config = _take(24)
        obj.tx_meas_signal_type = _take(1)
        obj.tx_meas_signal1_len = _take(3)
        obj.tx_meas_signal1_security = _take(2)
        obj.tx_meas_signal2_multitone = _take(2)
        obj.tx_first_sub_signal_len = _take(8)
        obj.tx_antenna_switch_interval = _take(8)
        obj.tx_meas_signal2_len = _take(8)
        obj.hop_band_bitmap = _take(8)
        if obj.hop_band_bitmap & 0x01:
            obj.hop_channel_2g4 = _take(80)
        if obj.hop_band_bitmap & 0x02:
            obj.hop_channel_5g1 = _take(200)
        if obj.hop_band_bitmap & 0x04:
            obj.hop_channel_5g8 = _take(128)
        obj.stability = _take(1)
        obj.reserved = _take(4)
        return obj


# ---------------------------------------------------------------------------
# 7.1.4.11 非链接态超宽带脉冲测量信息配置
# ---------------------------------------------------------------------------


@dataclass
class UWBPulseMeasurementConfig:
    """非链接态超宽带脉冲测量信息配置 (7.1.4.11, 表41)"""
    config_index: int = 0                     # 测量信号配置索引 (8b)
    event_group_start_offset: int = 0         # 事件组起始偏移时间 (24b, us)
    init_event_group_period: int = 0          # 初始化阶段事件组周期 (32b, 调度时隙)
    schedule_slot: int = 4                    # 系统调度时隙 (3b, 默认125us)
    init_channel_count: int = 0               # 初始化阶段的频点个数 (5b)
    init_channel_list: list[int] = field(default_factory=list)  # 频点列表 (每个2字节)
    init_interaction_type: int = 0            # 初始化阶段交互类型指示 (3b)
    init_sync_signal: int = 0                 # 初始化阶段同步信号指示 (5b)
    init_sync_config: int = 0                 # 初始化阶段同步信号配置 (24b)
    init_intra_event_interval: int = 0        # 初始化阶段事件内间隔 (16b, us)
    init_switch_interval: int = 0             # 初始化阶段测量帧内部切换间隔 (8b, us)
    init_meas_signal_type: int = 0            # 初始化阶段测量信号类型指示 (2b)
    init_meas_signal1_len: int = 0            # 初始化阶段测量信号1的长度 (6b)
    init_meas_signal2_len: int = 0            # 初始化阶段测量信号2的长度 (8b)
    tx_antenna_count: int = 0                 # 先发节点发送天线数量指示 (4b)
    tx_multi_antenna_config: int = 0          # 先发节点多天线切换资源配置指示 (8b)
    uwb_first_frame_start: int = 0            # 超宽带脉冲第一测量帧起始时刻 (22b)
    uwb_event_period: int = 0                 # 超宽带脉冲测量事件周期 (16b)
    uwb_event_group_period: int = 0           # 超宽带脉冲测量事件组周期 (16b, us)
    uwb_events_in_group: int = 0              # 超宽带脉冲测量事件组内事件总数 (8b)
    uwb_event_group_count: int = 0            # 超宽带脉冲测量事件组总数 (16b)
    uwb_event_stat_count: int = 0             # 超宽带脉冲测量事件统计数量 (8b)
    uwb_channel: int = 0                      # 超宽带脉冲信道指示 (8b)
    uwb_signal_bw: int = 0                    # 超宽带脉冲信号带宽指示 (2b)
    channel_overlap_mode: int = 0             # 信道重叠模式指示 (4b)
    channel_order_mode: int = 0               # 信道使用顺序模式指示 (2b)
    channel_stitch_count: int = 0             # 信道拼接数量指示 (8b)
    channel_stitch_step: int = 0              # 信道拼接测量子片段步进指示 (4b)
    sync_symbol_kl: int = 0                   # 同步符号码字长度K和占空因子L (4b)
    sync_symbol_n: int = 0                    # 同步符号个数N (16b)
    sync_symbol_index: int = 0                # 同步符号码字的索引 (8b)
    meas_symbol_shift: int = 0                # 测量符号码字循环移位 (8b)
    meas_seg_symbol_count: int = 0            # 测量子片段符号数Nseg (16b)
    meas_seg_count: int = 0                   # 测量子片段数Mseg (8b)
    meas_seg_gap: int = 0                     # 测量子片段间隔粒度Ngap (16b)
    meas_symbol_cp_len: int = 0               # 测量符号循环前缀长度Lcp (6b)
    meas_symbol_zero_len: int = 0             # 测量符号补零长度Lzero (8b)
    stability: int = 0                        # 稳定性指示 (1b)
    device_status: int = 0                    # 感知设备状态信息 (7b)

    def pack(self) -> bytes:
        bits: list[int] = []
        bits.extend(_int_to_bits(self.config_index, 8))
        bits.extend(_int_to_bits(self.event_group_start_offset, 24))
        bits.extend(_int_to_bits(self.init_event_group_period, 32))
        bits.extend(_int_to_bits(self.schedule_slot, 3))
        bits.extend(_int_to_bits(self.init_channel_count, 5))
        # 频点列表: 每个频点 16 bits
        for i in range(self.init_channel_count):
            ch = self.init_channel_list[i] if i < len(self.init_channel_list) else 0
            bits.extend(_int_to_bits(ch, 16))
        bits.extend(_int_to_bits(self.init_interaction_type, 3))
        bits.extend(_int_to_bits(self.init_sync_signal, 5))
        bits.extend(_int_to_bits(self.init_sync_config, 24))
        bits.extend(_int_to_bits(self.init_intra_event_interval, 16))
        bits.extend(_int_to_bits(self.init_switch_interval, 8))
        bits.extend(_int_to_bits(self.init_meas_signal_type, 2))
        bits.extend(_int_to_bits(self.init_meas_signal1_len, 6))
        bits.extend(_int_to_bits(self.init_meas_signal2_len, 8))
        bits.extend(_int_to_bits(self.tx_antenna_count, 4))
        bits.extend(_int_to_bits(self.tx_multi_antenna_config, 8))
        bits.extend(_int_to_bits(self.uwb_first_frame_start, 22))
        bits.extend(_int_to_bits(self.uwb_event_period, 16))
        bits.extend(_int_to_bits(self.uwb_event_group_period, 16))
        bits.extend(_int_to_bits(self.uwb_events_in_group, 8))
        bits.extend(_int_to_bits(self.uwb_event_group_count, 16))
        bits.extend(_int_to_bits(self.uwb_event_stat_count, 8))
        bits.extend(_int_to_bits(self.uwb_channel, 8))
        bits.extend(_int_to_bits(self.uwb_signal_bw, 2))
        bits.extend(_int_to_bits(self.channel_overlap_mode, 4))
        bits.extend(_int_to_bits(self.channel_order_mode, 2))
        bits.extend(_int_to_bits(self.channel_stitch_count, 8))
        bits.extend(_int_to_bits(self.channel_stitch_step, 4))
        bits.extend(_int_to_bits(self.sync_symbol_kl, 4))
        bits.extend(_int_to_bits(self.sync_symbol_n, 16))
        bits.extend(_int_to_bits(self.sync_symbol_index, 8))
        bits.extend(_int_to_bits(self.meas_symbol_shift, 8))
        bits.extend(_int_to_bits(self.meas_seg_symbol_count, 16))
        bits.extend(_int_to_bits(self.meas_seg_count, 8))
        bits.extend(_int_to_bits(self.meas_seg_gap, 16))
        bits.extend(_int_to_bits(self.meas_symbol_cp_len, 6))
        bits.extend(_int_to_bits(self.meas_symbol_zero_len, 8))
        bits.extend(_int_to_bits(self.stability, 1))
        bits.extend(_int_to_bits(self.device_status, 7))
        # 补齐到字节边界
        pad = (8 - len(bits) % 8) % 8
        bits.extend([0] * pad)
        return _bits_to_bytes(bits)

    @classmethod
    def unpack(cls, data: bytes) -> UWBPulseMeasurementConfig:
        bits = _bytes_to_bits(data)
        p = 0

        def _take(w: int) -> int:
            nonlocal p
            v = _bits_to_int(bits, p, w)
            p += w
            return v

        obj = cls()
        obj.config_index = _take(8)
        obj.event_group_start_offset = _take(24)
        obj.init_event_group_period = _take(32)
        obj.schedule_slot = _take(3)
        obj.init_channel_count = _take(5)
        obj.init_channel_list = [_take(16) for _ in range(obj.init_channel_count)]
        obj.init_interaction_type = _take(3)
        obj.init_sync_signal = _take(5)
        obj.init_sync_config = _take(24)
        obj.init_intra_event_interval = _take(16)
        obj.init_switch_interval = _take(8)
        obj.init_meas_signal_type = _take(2)
        obj.init_meas_signal1_len = _take(6)
        obj.init_meas_signal2_len = _take(8)
        obj.tx_antenna_count = _take(4)
        obj.tx_multi_antenna_config = _take(8)
        obj.uwb_first_frame_start = _take(22)
        obj.uwb_event_period = _take(16)
        obj.uwb_event_group_period = _take(16)
        obj.uwb_events_in_group = _take(8)
        obj.uwb_event_group_count = _take(16)
        obj.uwb_event_stat_count = _take(8)
        obj.uwb_channel = _take(8)
        obj.uwb_signal_bw = _take(2)
        obj.channel_overlap_mode = _take(4)
        obj.channel_order_mode = _take(2)
        obj.channel_stitch_count = _take(8)
        obj.channel_stitch_step = _take(4)
        obj.sync_symbol_kl = _take(4)
        obj.sync_symbol_n = _take(16)
        obj.sync_symbol_index = _take(8)
        obj.meas_symbol_shift = _take(8)
        obj.meas_seg_symbol_count = _take(16)
        obj.meas_seg_count = _take(8)
        obj.meas_seg_gap = _take(16)
        obj.meas_symbol_cp_len = _take(6)
        obj.meas_symbol_zero_len = _take(8)
        obj.stability = _take(1)
        obj.device_status = _take(7)
        return obj


# ---------------------------------------------------------------------------
# 比特操作辅助函数
# ---------------------------------------------------------------------------


def _int_to_bits(val: int, width: int) -> list[int]:
    """整数转比特列表 (高位在前)"""
    return [(val >> (width - 1 - i)) & 1 for i in range(width)]


def _bits_to_int(bits: list[int], start: int, width: int) -> int:
    """比特列表转整数"""
    val = 0
    for i in range(width):
        if start + i < len(bits):
            val = (val << 1) | (bits[start + i] & 1)
        else:
            val <<= 1
    return val


def _bits_to_bytes(bits: list[int]) -> bytes:
    """比特列表转字节序列"""
    result = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val = (byte_val << 1) | (bits[i + j] & 1)
            else:
                byte_val <<= 1
        result.append(byte_val)
    return bytes(result)


def _bytes_to_bits(data: bytes) -> list[int]:
    """字节序列转比特列表"""
    bits: list[int] = []
    bits.extend((b >> i) & 1 for b in data for i in range(7, -1, -1))
    return bits
