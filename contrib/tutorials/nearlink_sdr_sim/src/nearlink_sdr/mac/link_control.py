"""链路控制信令 -- TXS-10002-2025 标准 7.3.2.2-7.3.2.32

提供链路建立、参数协商、断开等核心控制面信令的编解码。
"""

from __future__ import annotations

__all__ = [
    "AsyncLinkParamRequest",
    "AsyncLinkParamResponse",
    "AsyncMulticastLinkSetup",
    "AsyncMulticastParamExchangeRequest",
    "AsyncMulticastParamExchangeResponse",
    "AsyncMulticastParamUpdateIndication",
    "AsyncMulticastParamUpdateRequest",
    "AsyncMulticastReconfig",
    "AsyncTTLinkSetup",
    "AsyncUnicastUpdate",
    "BroadcastHopMap5GUpdate",
    "BroadcastHopMapUpdate",
    "BroadcastLinkDisconnect",
    "BroadcastLinkParamUpdate",
    "BroadcastLinkSetup",
    "Channel5GStatusIndication",
    "ChannelReportConfig",
    "ChannelStatusIndication",
    "ClockAccuracyRequest",
    "ClockAccuracyResponse",
    "CoordinateConfig",
    "CoordinateReport",
    "CoordinateRequest",
    "CrcSwitchIndication",
    "CrcSwitchRequest",
    "DataLengthRequest",
    "DataLengthResponse",
    "FeatureExchangeRequest",
    "FeatureExchangeResponse",
    "HopMap5GUpdate",
    "HopMapUpdate",
    "HopTableUpdate",
    "IntervalUpdateIndication",
    "IntervalUpdateRequest",
    "IntervalUpdateResponse",
    "IsochronousLinkSetup",
    "IsochronousParamExchangeRequest",
    "IsochronousParamExchangeResponse",
    "IsochronousParamUpdateIndication",
    "IsochronousParamUpdateRequest",
    "LinkDisconnect",
    "MinAvailableChannels",
    "MultiIntervalUpdateIndication",
    "MultiIntervalUpdateRequest",
    "MultiIntervalUpdateResponse",
    "MulticastDisconnect",
    "NarrowbandDelayRequest",
    "NarrowbandDelayResponse",
    "NarrowbandFreqTable24Update",
    "NarrowbandFreqTable51Update",
    "NarrowbandFreqTable58Update",
    "NarrowbandMeasAction",
    "NarrowbandMeasCapRequest",
    "NarrowbandMeasCapResponse",
    "NarrowbandMeasConfig",
    "NarrowbandMeasConfigUpdateIndication",
    "NarrowbandMeasConfigUpdateRequest",
    "NarrowbandMeasReport",
    "NarrowbandProxySensingFeedback",
    "NarrowbandProxySensingRequest",
    "NarrowbandSensingAction",
    "NarrowbandSensingCapRequest",
    "NarrowbandSensingCapResponse",
    "NarrowbandSensingConfig",
    "NarrowbandSensingConfigFeedback",
    "NarrowbandSensingFeedback",
    "NarrowbandSensingReport",
    "NarrowbandSensingRequest",
    "PhyUpdateIndication",
    "PhyUpdateRequest",
    "PingRequest",
    "PingResponse",
    "ResourceReservation",
    "ResourceReservationTerminate",
    "RoleSwitchRequest",
    "SMFParamUpdateIndication",
    "SMFParamUpdateRequest",
    "SMFSignalingTerminate",
    "SMFTimeSlotUpdateRequest",
    "SMFTimeSlotUpdateResponse",
    "SecurityPauseRequest",
    "SecurityPauseResponse",
    "SecurityRequest",
    "SecurityResponse",
    "SecurityStartRequest",
    "SecurityStartResponse",
    "SensingDeviceStatusReport",
    "SignalingReject",
    "SystemTimeIndication",
    "TimeOffsetIndication",
    "TimeoutUpdateRequest",
    "UWBMeasAction",
    "UWBMeasCapRequest",
    "UWBMeasCapResponse",
    "UWBMeasConfig",
    "UWBMeasConfigFeedback",
    "UWBMeasReport",
    "UWBProxySensingFeedback",
    "UWBProxySensingRequest",
    "UWBSensingAction",
    "UWBSensingCapRequest",
    "UWBSensingCapResponse",
    "UWBSensingConfig",
    "UWBSensingConfigFeedback",
    "UWBSensingProcessFeedback",
    "UWBSensingProcessRequest",
    "UWBSensingReport",
    "UnknownFeatureFeedback",
    "VersionExchange",
]


import struct
from dataclasses import dataclass, fields

# ---------------------------------------------------------------------------
# 7.3.2.2 收发间隔更新请求 (0x0000, 8 bits / 1 byte)
# ---------------------------------------------------------------------------

@dataclass
class IntervalUpdateRequest:
    """收发间隔更新请求。

    字段:
    - interval_type: 4 bits, 请求的收发间隔类型 (0-15)
    - reserved:      4 bits
    """
    interval_type: int  # 0-15

    DATA_TYPE_INDEX = 0x0000
    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        return bytes([(self.interval_type & 0x0F) << 4])

    @classmethod
    def unpack(cls, data: bytes) -> IntervalUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(interval_type=(data[0] >> 4) & 0x0F)


# ---------------------------------------------------------------------------
# 7.3.2.3 收发间隔更新响应 (0x0001, 8 bits / 1 byte)
# ---------------------------------------------------------------------------

@dataclass
class IntervalUpdateResponse:
    """收发间隔更新响应。

    字段:
    - interval_type: 4 bits, 响应的收发间隔类型 (0-15)
    - reserved:      4 bits
    """
    interval_type: int  # 0-15

    DATA_TYPE_INDEX = 0x0001
    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        return bytes([(self.interval_type & 0x0F) << 4])

    @classmethod
    def unpack(cls, data: bytes) -> IntervalUpdateResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(interval_type=(data[0] >> 4) & 0x0F)


# ---------------------------------------------------------------------------
# 7.3.2.4 收发间隔更新指示 (0x0002, 64 bits / 8 bytes)
# ---------------------------------------------------------------------------

@dataclass
class IntervalUpdateIndication:
    """收发间隔更新指示。

    字段:
    - link_id:           24 bits, 逻辑链路标识
    - interval_type:     4 bits, 更新后的收发间隔类型
    - reserved:          4 bits
    - effective_slot:    32 bits, 信令生效时隙号
    """
    link_id: int           # 1 ~ 2^24-1
    interval_type: int     # 0-15
    effective_slot: int    # 0 ~ 2^30-1

    DATA_TYPE_INDEX = 0x0002
    BYTE_LENGTH = 8

    def pack(self) -> bytes:
        b = self.link_id.to_bytes(3, "big")
        b += bytes([(self.interval_type & 0x0F) << 4])
        b += self.effective_slot.to_bytes(4, "big")
        return b

    @classmethod
    def unpack(cls, data: bytes) -> IntervalUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        link_id = int.from_bytes(data[0:3], "big")
        interval_type = (data[3] >> 4) & 0x0F
        effective_slot = int.from_bytes(data[4:8], "big")
        return cls(link_id, interval_type, effective_slot)


# ---------------------------------------------------------------------------
# 7.3.2.5 信令被拒指示 (0x0003, 24 bits / 3 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SignalingReject:
    """信令被拒指示。

    字段:
    - rejected_index: 16 bits, 被拒的数据类型索引
    - error_reason:    8 bits, 出错原因
    """
    rejected_index: int  # 0-65535
    error_reason: int    # 0-255

    DATA_TYPE_INDEX = 0x0003
    BYTE_LENGTH = 3

    def pack(self) -> bytes:
        return self.rejected_index.to_bytes(2, "big") + bytes([self.error_reason & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> SignalingReject:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        rejected_index = int.from_bytes(data[0:2], "big")
        return cls(rejected_index, data[2])


# ---------------------------------------------------------------------------
# 7.3.2.12 特性交互请求 (0x000A, 80 bits / 10 bytes)
# ---------------------------------------------------------------------------

@dataclass
class FeatureExchangeRequest:
    """特性交互请求: 80-bit 特性集位图。"""
    feature_set: int  # 0 ~ 2^80-1

    DATA_TYPE_INDEX = 0x000A
    BYTE_LENGTH = 10

    def pack(self) -> bytes:
        return self.feature_set.to_bytes(10, "big")

    @classmethod
    def unpack(cls, data: bytes) -> FeatureExchangeRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:10], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.13 特性交互响应 (0x000B, 80 bits / 10 bytes)
# ---------------------------------------------------------------------------

@dataclass
class FeatureExchangeResponse:
    """特性交互响应: 80-bit 特性集位图。"""
    feature_set: int  # 0 ~ 2^80-1

    DATA_TYPE_INDEX = 0x000B
    BYTE_LENGTH = 10

    def pack(self) -> bytes:
        return self.feature_set.to_bytes(10, "big")

    @classmethod
    def unpack(cls, data: bytes) -> FeatureExchangeResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:10], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.15 版本交互指示 (0x000D, 40 bits / 5 bytes)
# ---------------------------------------------------------------------------

@dataclass
class VersionExchange:
    """版本交互指示。

    字段:
    - spec_version:     8 bits, 规格版本
    - company_id:      16 bits, 公司标识符
    - sub_version:     16 bits, 子规格版本
    """
    spec_version: int   # 0-255
    company_id: int     # 0-65535
    sub_version: int    # 0-65535

    DATA_TYPE_INDEX = 0x000D
    BYTE_LENGTH = 5

    def pack(self) -> bytes:
        return (bytes([self.spec_version & 0xFF])
                + self.company_id.to_bytes(2, "big")
                + self.sub_version.to_bytes(2, "big"))

    @classmethod
    def unpack(cls, data: bytes) -> VersionExchange:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], int.from_bytes(data[1:3], "big"),
                   int.from_bytes(data[3:5], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.16 数据长度请求 (0x000E, 64 bits / 8 bytes)
# ---------------------------------------------------------------------------

@dataclass
class DataLengthRequest:
    """数据长度请求 (MTU 协商)。

    字段:
    - max_rx_bytes: 16 bits, 最大接收字节数 (31-2047)
    - max_rx_time:  16 bits, 最大接收时间 (346-65535 us)
    - max_tx_bytes: 16 bits, 最大发送字节数 (31-2047)
    - max_tx_time:  16 bits, 最大发送时间 (346-65535 us)
    """
    max_rx_bytes: int
    max_rx_time: int
    max_tx_bytes: int
    max_tx_time: int

    DATA_TYPE_INDEX = 0x000E
    BYTE_LENGTH = 8

    def pack(self) -> bytes:
        return struct.pack(">HHHH", self.max_rx_bytes, self.max_rx_time,
                           self.max_tx_bytes, self.max_tx_time)

    @classmethod
    def unpack(cls, data: bytes) -> DataLengthRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        vals = struct.unpack(">HHHH", data[:8])
        return cls(*vals)


# ---------------------------------------------------------------------------
# 7.3.2.17 数据长度响应 (0x000F, 64 bits / 8 bytes)
# ---------------------------------------------------------------------------

@dataclass
class DataLengthResponse:
    """数据长度响应。字段与 DataLengthRequest 相同。"""
    max_rx_bytes: int
    max_rx_time: int
    max_tx_bytes: int
    max_tx_time: int

    DATA_TYPE_INDEX = 0x000F
    BYTE_LENGTH = 8

    def pack(self) -> bytes:
        return struct.pack(">HHHH", self.max_rx_bytes, self.max_rx_time,
                           self.max_tx_bytes, self.max_tx_time)

    @classmethod
    def unpack(cls, data: bytes) -> DataLengthResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        vals = struct.unpack(">HHHH", data[:8])
        return cls(*vals)


# ---------------------------------------------------------------------------
# 7.3.2.18 信道上报指示 (0x0010, 24 bits / 3 bytes)
# ---------------------------------------------------------------------------

@dataclass
class ChannelReportConfig:
    """信道上报指示。

    字段:
    - enable:       8 bits, 使能信道上报 (0 或 1)
    - min_interval: 8 bits, 最小时间间隔 (5-150, 单位 200ms)
    - max_delay:    8 bits, 最大时延 (5-150, 单位 200ms)
    """
    enable: int        # 0 或 1
    min_interval: int  # 5-150
    max_delay: int     # 5-150

    DATA_TYPE_INDEX = 0x0010
    BYTE_LENGTH = 3

    def pack(self) -> bytes:
        return bytes([self.enable & 0xFF, self.min_interval & 0xFF,
                      self.max_delay & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> ChannelReportConfig:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1], data[2])


# ---------------------------------------------------------------------------
# 7.3.2.23 CRC 切换请求 (0x0015, 96 bits / 12 bytes)
# ---------------------------------------------------------------------------

@dataclass
class CrcSwitchRequest:
    """CRC 切换请求。

    字段:
    - link_id:         24 bits, 逻辑链路标识
    - tx_crc_type:      1 bit, 先发链路 CRC 类型 (0=CRC24, 1=CRC32)
    - rx_crc_type:      1 bit, 后发链路 CRC 类型
    - reserved:         6 bits
    - tx_crc_init:     32 bits, 先发链路 CRC 初始值
    - rx_crc_init:     32 bits, 后发链路 CRC 初始值
    """
    link_id: int
    tx_crc_type: int     # 0=CRC24, 1=CRC32
    rx_crc_type: int     # 0=CRC24, 1=CRC32
    tx_crc_init: int     # CRC 初始值
    rx_crc_init: int     # CRC 初始值

    DATA_TYPE_INDEX = 0x0015
    BYTE_LENGTH = 12

    def pack(self) -> bytes:
        b = self.link_id.to_bytes(3, "big")
        flags = ((self.tx_crc_type & 1) << 7) | ((self.rx_crc_type & 1) << 6)
        b += bytes([flags])
        b += self.tx_crc_init.to_bytes(4, "big")
        b += self.rx_crc_init.to_bytes(4, "big")
        return b

    @classmethod
    def unpack(cls, data: bytes) -> CrcSwitchRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        link_id = int.from_bytes(data[0:3], "big")
        tx_crc_type = (data[3] >> 7) & 1
        rx_crc_type = (data[3] >> 6) & 1
        tx_crc_init = int.from_bytes(data[4:8], "big")
        rx_crc_init = int.from_bytes(data[8:12], "big")
        return cls(link_id, tx_crc_type, rx_crc_type, tx_crc_init, rx_crc_init)


# ---------------------------------------------------------------------------
# 7.3.2.24 CRC 切换指示 (0x0016, 128 bits / 16 bytes)
# ---------------------------------------------------------------------------

@dataclass
class CrcSwitchIndication:
    """CRC 切换指示。增加生效时隙号。"""
    link_id: int
    tx_crc_type: int
    rx_crc_type: int
    tx_crc_init: int
    rx_crc_init: int
    effective_slot: int  # 32 bits

    DATA_TYPE_INDEX = 0x0016
    BYTE_LENGTH = 16

    def pack(self) -> bytes:
        b = self.link_id.to_bytes(3, "big")
        flags = ((self.tx_crc_type & 1) << 7) | ((self.rx_crc_type & 1) << 6)
        b += bytes([flags])
        b += self.tx_crc_init.to_bytes(4, "big")
        b += self.rx_crc_init.to_bytes(4, "big")
        b += self.effective_slot.to_bytes(4, "big")
        return b

    @classmethod
    def unpack(cls, data: bytes) -> CrcSwitchIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        link_id = int.from_bytes(data[0:3], "big")
        tx_crc_type = (data[3] >> 7) & 1
        rx_crc_type = (data[3] >> 6) & 1
        tx_crc_init = int.from_bytes(data[4:8], "big")
        rx_crc_init = int.from_bytes(data[8:12], "big")
        effective_slot = int.from_bytes(data[12:16], "big")
        return cls(link_id, tx_crc_type, rx_crc_type,
                   tx_crc_init, rx_crc_init, effective_slot)


# ---------------------------------------------------------------------------
# 7.3.2.30 时钟精度请求 (0x001C, 8 bits / 1 byte)
# ---------------------------------------------------------------------------

@dataclass
class ClockAccuracyRequest:
    """时钟精度请求。"""
    accuracy: int  # 0-255, 枚举

    DATA_TYPE_INDEX = 0x001C
    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        return bytes([self.accuracy & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> ClockAccuracyRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0])


# ---------------------------------------------------------------------------
# 7.3.2.31 时钟精度响应 (0x001D, 8 bits / 1 byte)
# ---------------------------------------------------------------------------

@dataclass
class ClockAccuracyResponse:
    """时钟精度响应。"""
    accuracy: int  # 0-255

    DATA_TYPE_INDEX = 0x001D
    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        return bytes([self.accuracy & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> ClockAccuracyResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0])


# ---------------------------------------------------------------------------
# 7.3.2.32 链路断开指示 (0x001E, 32 bits / 4 bytes)
# ---------------------------------------------------------------------------

@dataclass
class LinkDisconnect:
    """链路断开指示。

    字段:
    - link_id:      24 bits, 逻辑链路标识
    - error_reason:  8 bits, 出错原因
    """
    link_id: int      # 0 ~ 2^24-1
    error_reason: int  # 0-255

    DATA_TYPE_INDEX = 0x001E
    BYTE_LENGTH = 4

    def pack(self) -> bytes:
        return self.link_id.to_bytes(3, "big") + bytes([self.error_reason & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> LinkDisconnect:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        link_id = int.from_bytes(data[0:3], "big")
        return cls(link_id, data[3])


# ---------------------------------------------------------------------------
# 7.3.2.6 安全请求 (0x0004, 104 bits / 13 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SecurityRequest:
    """安全请求。

    字段:
    - g_node_iv:       32 bits, G节点初始化向量
    - g_node_skd:      64 bits, G节点连接密钥分散器
    - enc_indication:   8 bits, 加密和完整性保护指示
    """
    g_node_iv: int
    g_node_skd: int
    enc_indication: int

    DATA_TYPE_INDEX = 0x0004
    BYTE_LENGTH = 13

    def pack(self) -> bytes:
        return (self.g_node_iv.to_bytes(4, "big")
                + self.g_node_skd.to_bytes(8, "big")
                + bytes([self.enc_indication & 0xFF]))

    @classmethod
    def unpack(cls, data: bytes) -> SecurityRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        g_iv = int.from_bytes(data[0:4], "big")
        g_skd = int.from_bytes(data[4:12], "big")
        return cls(g_iv, g_skd, data[12])


# ---------------------------------------------------------------------------
# 7.3.2.7 安全响应 (0x0005, 96 bits / 12 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SecurityResponse:
    """安全响应。

    字段:
    - t_node_iv:  32 bits, T节点初始化向量
    - t_node_skd: 64 bits, T节点连接密钥分散器
    """
    t_node_iv: int
    t_node_skd: int

    DATA_TYPE_INDEX = 0x0005
    BYTE_LENGTH = 12

    def pack(self) -> bytes:
        return (self.t_node_iv.to_bytes(4, "big")
                + self.t_node_skd.to_bytes(8, "big"))

    @classmethod
    def unpack(cls, data: bytes) -> SecurityResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[0:4], "big"),
                   int.from_bytes(data[4:12], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.8 安全启动请求 (0x0006, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SecurityStartRequest:
    """安全启动请求 (无载荷)。"""

    DATA_TYPE_INDEX = 0x0006
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> SecurityStartRequest:
        return cls()


# ---------------------------------------------------------------------------
# 7.3.2.9 安全启动响应 (0x0007, 8 bits / 1 byte)
# ---------------------------------------------------------------------------

@dataclass
class SecurityStartResponse:
    """安全启动响应。

    字段:
    - enc_indication: 8 bits, 加密和完整性保护指示
    """
    enc_indication: int

    DATA_TYPE_INDEX = 0x0007
    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        return bytes([self.enc_indication & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> SecurityStartResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0])


# ---------------------------------------------------------------------------
# 7.3.2.10 安全暂停请求 (0x0008, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SecurityPauseRequest:
    """安全暂停请求 (无载荷)。"""

    DATA_TYPE_INDEX = 0x0008
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> SecurityPauseRequest:
        return cls()


# ---------------------------------------------------------------------------
# 7.3.2.11 安全暂停响应 (0x0009, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SecurityPauseResponse:
    """安全暂停响应 (无载荷)。"""

    DATA_TYPE_INDEX = 0x0009
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> SecurityPauseResponse:
        return cls()


# ---------------------------------------------------------------------------
# 7.3.2.14 未知特性反馈 (0x000C, 16 bits / 2 bytes)
# ---------------------------------------------------------------------------

@dataclass
class UnknownFeatureFeedback:
    """未知特性反馈。

    字段:
    - unknown_type: 16 bits, 收到的未知数据类型索引
    """
    unknown_type: int

    DATA_TYPE_INDEX = 0x000C
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return self.unknown_type.to_bytes(2, "big")

    @classmethod
    def unpack(cls, data: bytes) -> UnknownFeatureFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:2], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.19 信道状态指示 (0x0011, 160 bits / 20 bytes)
# ---------------------------------------------------------------------------

@dataclass
class ChannelStatusIndication:
    """信道状态指示。

    字段:
    - channel_map: 160 bits, 每信道 2 bit 质量指示 (0=未知, 1=好, 3=差)
    """
    channel_map: bytes  # 20 bytes

    DATA_TYPE_INDEX = 0x0011
    BYTE_LENGTH = 20

    def pack(self) -> bytes:
        return self.channel_map[:20].ljust(20, b"\x00")

    @classmethod
    def unpack(cls, data: bytes) -> ChannelStatusIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:20]))


# ---------------------------------------------------------------------------
# 7.3.2.20 跳频表更新指示 (0x0012, 可变长度)
# ---------------------------------------------------------------------------

@dataclass
class HopTableUpdate:
    """跳频表更新指示 (系统管理帧)。

    字段:
    - effective_slot: 32 bits, 信令生效时隙号
    - channel_count:   8 bits, 跳频表频点个数
    - channel_table:  可变,  跳频表频点列表
    """
    effective_slot: int
    channel_count: int
    channel_table: bytes

    DATA_TYPE_INDEX = 0x0012
    BYTE_LENGTH = 5  # 最小长度

    def pack(self) -> bytes:
        return (self.effective_slot.to_bytes(4, "big")
                + bytes([self.channel_count & 0xFF])
                + self.channel_table)

    @classmethod
    def unpack(cls, data: bytes) -> HopTableUpdate:
        if len(data) < 5:
            raise ValueError("数据不足: 至少需要 5 字节")
        slot = int.from_bytes(data[0:4], "big")
        count = data[4]
        table = bytes(data[5:5 + count])
        return cls(slot, count, table)


# ---------------------------------------------------------------------------
# 7.3.2.21 跳频地图更新指示 (0x0013, 112 bits / 14 bytes)
# ---------------------------------------------------------------------------

@dataclass
class HopMapUpdate:
    """跳频地图更新指示 (数据链路)。

    字段:
    - hop_map:        80 bits, 跳频地图位图
    - effective_slot: 32 bits, 信令生效时隙号
    """
    hop_map: bytes         # 10 bytes
    effective_slot: int

    DATA_TYPE_INDEX = 0x0013
    BYTE_LENGTH = 14

    def pack(self) -> bytes:
        return self.hop_map[:10].ljust(10, b"\x00") + self.effective_slot.to_bytes(4, "big")

    @classmethod
    def unpack(cls, data: bytes) -> HopMapUpdate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:10]), int.from_bytes(data[10:14], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.22 最少可用信道指示 (0x0014, 16 bits / 2 bytes)
# ---------------------------------------------------------------------------

@dataclass
class MinAvailableChannels:
    """最少可用信道指示。

    字段:
    - frame_type:     4 bits, 无线帧类型
    - bandwidth:      2 bits, 带宽指示
    - pilot_density:  2 bits, 导频密度指示
    - min_channels:   8 bits, 最小信道数 (2-76)
    """
    frame_type: int
    bandwidth: int
    pilot_density: int
    min_channels: int

    DATA_TYPE_INDEX = 0x0014
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        b0 = ((self.frame_type & 0x0F) << 4
              | (self.bandwidth & 0x03) << 2
              | (self.pilot_density & 0x03))
        return bytes([b0, self.min_channels & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> MinAvailableChannels:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        ft = (data[0] >> 4) & 0x0F
        bw = (data[0] >> 2) & 0x03
        pd = data[0] & 0x03
        return cls(ft, bw, pd, data[1])


# ---------------------------------------------------------------------------
# 7.3.2.25 物理层更新请求 (0x0017, 32 bits / 4 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PhyUpdateRequest:
    """物理层更新请求。

    字段:
    - tx_frame_type:      4 bits, 先发链路无线帧类型
    - rx_frame_type:      4 bits, 后发链路无线帧类型
    - tx_bandwidth:       2 bits, 先发链路带宽
    - rx_bandwidth:       2 bits, 后发链路带宽
    - tx_pilot_density:   2 bits, 先发链路导频密度
    - rx_pilot_density:   2 bits, 后发链路导频密度
    - tx_feedback_type:   6 bits, 先发链路反馈类型
    - rx_feedback_type:   3 bits, 后发链路反馈类型
    - reserved:           7 bits
    """
    tx_frame_type: int
    rx_frame_type: int
    tx_bandwidth: int
    rx_bandwidth: int
    tx_pilot_density: int
    rx_pilot_density: int
    tx_feedback_type: int
    rx_feedback_type: int

    DATA_TYPE_INDEX = 0x0017
    BYTE_LENGTH = 4

    def pack(self) -> bytes:
        b0 = ((self.tx_frame_type & 0x0F) << 4) | (self.rx_frame_type & 0x0F)
        b1 = ((self.tx_bandwidth & 0x03) << 6
              | (self.rx_bandwidth & 0x03) << 4
              | (self.tx_pilot_density & 0x03) << 2
              | (self.rx_pilot_density & 0x03))
        b2 = ((self.tx_feedback_type & 0x3F) << 2
              | (self.rx_feedback_type >> 1) & 0x03)
        b3 = ((self.rx_feedback_type & 0x01) << 7)
        return bytes([b0, b1, b2, b3])

    @classmethod
    def unpack(cls, data: bytes) -> PhyUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        tx_ft = (data[0] >> 4) & 0x0F
        rx_ft = data[0] & 0x0F
        tx_bw = (data[1] >> 6) & 0x03
        rx_bw = (data[1] >> 4) & 0x03
        tx_pd = (data[1] >> 2) & 0x03
        rx_pd = data[1] & 0x03
        tx_fb = (data[2] >> 2) & 0x3F
        rx_fb = ((data[2] & 0x03) << 1) | ((data[3] >> 7) & 0x01)
        return cls(tx_ft, rx_ft, tx_bw, rx_bw, tx_pd, rx_pd, tx_fb, rx_fb)


# ---------------------------------------------------------------------------
# 7.3.2.26 物理层更新指示 (0x0018, 64 bits / 8 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PhyUpdateIndication:
    """物理层更新指示 (含生效时隙)。"""
    tx_frame_type: int
    rx_frame_type: int
    tx_bandwidth: int
    rx_bandwidth: int
    tx_pilot_density: int
    rx_pilot_density: int
    tx_feedback_type: int
    rx_feedback_type: int
    effective_slot: int

    DATA_TYPE_INDEX = 0x0018
    BYTE_LENGTH = 8

    def pack(self) -> bytes:
        phy_req = PhyUpdateRequest(
            self.tx_frame_type, self.rx_frame_type,
            self.tx_bandwidth, self.rx_bandwidth,
            self.tx_pilot_density, self.rx_pilot_density,
            self.tx_feedback_type, self.rx_feedback_type,
        )
        return phy_req.pack() + self.effective_slot.to_bytes(4, "big")

    @classmethod
    def unpack(cls, data: bytes) -> PhyUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        req = PhyUpdateRequest.unpack(data[:4])
        slot = int.from_bytes(data[4:8], "big")
        return cls(
            req.tx_frame_type, req.rx_frame_type,
            req.tx_bandwidth, req.rx_bandwidth,
            req.tx_pilot_density, req.rx_pilot_density,
            req.tx_feedback_type, req.rx_feedback_type, slot,
        )


# ---------------------------------------------------------------------------
# 7.3.2.33 异步组播链路参数重配置指示 (0x001F, 248 bits / 31 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncMulticastReconfig:
    """异步组播链路参数重配置指示。"""
    effective_ref_slot: int      # 32 bits
    event_group_offset: int      # 16 bits
    event_group_period: int      # 16 bits
    event_period: int            # 16 bits
    delay_period: int            # 16 bits
    timeout: int                 # 16 bits
    intra_event_interval: int    # 16 bits
    inter_event_interval: int    # 16 bits
    event_count: int             # 8 bits
    payload_count: int           # 39 bits
    scheduling_slot: int         # 3 bits
    tx_rx_indication: int        # 1 bit
    tx_max_pdu: int              # 11 bits
    rx_max_pdu: int              # 11 bits
    tx_max_time_offset: int      # 9 bits
    rx_max_time_offset: int      # 9 bits

    DATA_TYPE_INDEX = 0x001F
    BYTE_LENGTH = 30

    def pack(self) -> bytes:
        buf = self.effective_ref_slot.to_bytes(4, "big")
        buf += struct.pack(">HHHHHHHB",
                           self.event_group_offset, self.event_group_period,
                           self.event_period, self.delay_period,
                           self.timeout, self.intra_event_interval,
                           self.inter_event_interval, self.event_count)
        # payload_count(39) + reserved(5) + scheduling_slot(3) +
        # tx_rx_indication(1) + tx_max_pdu(11) + rx_max_pdu(11) +
        # tx_max_time_offset(9) + rx_max_time_offset(9) = 88 bits = 11 bytes
        pc = self.payload_count & 0x7FFFFFFFFF
        bits = (pc << 49
                | (self.scheduling_slot & 0x07) << 41
                | (self.tx_rx_indication & 0x01) << 40
                | (self.tx_max_pdu & 0x7FF) << 29
                | (self.rx_max_pdu & 0x7FF) << 18
                | (self.tx_max_time_offset & 0x1FF) << 9
                | (self.rx_max_time_offset & 0x1FF))
        buf += bits.to_bytes(11, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AsyncMulticastReconfig:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        efs = int.from_bytes(data[0:4], "big")
        ego, egp, ep, dp, to, iei, iei2, ec = struct.unpack(
            ">HHHHHHHB", data[4:19])
        tail = int.from_bytes(data[19:30], "big")
        pc = (tail >> 49) & 0x7FFFFFFFFF
        ss = (tail >> 41) & 0x07
        tri = (tail >> 40) & 0x01
        tmp = (tail >> 29) & 0x7FF
        rmp = (tail >> 18) & 0x7FF
        tmto = (tail >> 9) & 0x1FF
        rmto = tail & 0x1FF
        return cls(efs, ego, egp, ep, dp, to, iei, iei2, ec,
                   pc, ss, tri, tmp, rmp, tmto, rmto)


# ---------------------------------------------------------------------------
# 7.3.2.34 链接态异步链路参数更新指示 (0x003A, 120 bits / 15 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncUnicastUpdate:
    """链接态单播异步链路参数更新指示。

    标准定义 120 bits = 15 bytes。timeout 字段不包含在标准序列化中。
    """
    effective_ref_slot: int      # 32 bits
    event_group_offset: int      # 16 bits
    event_group_period: int      # 16 bits
    intra_event_interval: int    # 16 bits
    inter_event_interval: int    # 16 bits
    delay_period: int            # 16 bits
    timeout: int = 0             # 不参与序列化
    scheduling_slot: int = 0     # 3 bits
    tx_rx_indication: int = 0    # 1 bit

    DATA_TYPE_INDEX = 0x003A
    BYTE_LENGTH = 15

    def pack(self) -> bytes:
        buf = self.effective_ref_slot.to_bytes(4, "big")
        buf += struct.pack(">HHHHH",
                           self.event_group_offset, self.event_group_period,
                           self.intra_event_interval, self.inter_event_interval,
                           self.delay_period)
        tail = ((self.scheduling_slot & 0x07) << 5
                | (self.tx_rx_indication & 0x01) << 4)
        buf += bytes([tail])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AsyncUnicastUpdate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        efs = int.from_bytes(data[0:4], "big")
        ego = int.from_bytes(data[4:6], "big")
        egp = int.from_bytes(data[6:8], "big")
        iei = int.from_bytes(data[8:10], "big")
        iei2 = int.from_bytes(data[10:12], "big")
        dp = int.from_bytes(data[12:14], "big")
        tail = data[14]
        ss = (tail >> 5) & 0x07
        tri = (tail >> 4) & 0x01
        return cls(efs, ego, egp, iei, iei2, dp, 0, ss, tri)


# ---------------------------------------------------------------------------
# 7.3.2.45 广播链路断开指示 (0x002A, 40 bits / 5 bytes)
# ---------------------------------------------------------------------------

@dataclass
class BroadcastLinkDisconnect:
    """广播链路断开指示。

    字段:
    - link_id:      24 bits, 逻辑链路标识
    - error_reason:  8 bits, 出错原因
    - reserved:      8 bits
    """
    link_id: int
    error_reason: int

    DATA_TYPE_INDEX = 0x002A
    BYTE_LENGTH = 5

    def pack(self) -> bytes:
        return self.link_id.to_bytes(3, "big") + bytes([self.error_reason & 0xFF, 0])

    @classmethod
    def unpack(cls, data: bytes) -> BroadcastLinkDisconnect:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[0:3], "big"), data[3])


# ---------------------------------------------------------------------------
# 7.3.2.51 系统管理帧信令传输终止 (0x0030, 8 bits / 1 byte)
# ---------------------------------------------------------------------------

@dataclass
class SMFSignalingTerminate:
    """系统管理帧信令传输终止。"""
    terminate_type: int  # 8 bits

    DATA_TYPE_INDEX = 0x0030
    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        return bytes([self.terminate_type & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> SMFSignalingTerminate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0])


# ---------------------------------------------------------------------------
# 7.3.2.52 角色切换请求 (0x0031, 32 bits / 4 bytes)
# ---------------------------------------------------------------------------

@dataclass
class RoleSwitchRequest:
    """角色切换请求。

    字段:
    - effective_slot: 32 bits, 信令生效时隙号
    """
    effective_slot: int

    DATA_TYPE_INDEX = 0x0031
    BYTE_LENGTH = 4

    def pack(self) -> bytes:
        return self.effective_slot.to_bytes(4, "big")

    @classmethod
    def unpack(cls, data: bytes) -> RoleSwitchRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:4], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.53 时间偏移指示 (0x0032, 64 bits / 8 bytes)
# ---------------------------------------------------------------------------

@dataclass
class TimeOffsetIndication:
    """时间偏移指示。"""
    time_offset: int  # 64 bits

    DATA_TYPE_INDEX = 0x0032
    BYTE_LENGTH = 8

    def pack(self) -> bytes:
        return self.time_offset.to_bytes(8, "big")

    @classmethod
    def unpack(cls, data: bytes) -> TimeOffsetIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:8], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.54 PING 请求 (0x0033, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PingRequest:
    """PING 请求 (无载荷)。"""

    DATA_TYPE_INDEX = 0x0033
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> PingRequest:
        return cls()


# ---------------------------------------------------------------------------
# 7.3.2.55 PING 响应 (0x0034, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PingResponse:
    """PING 响应 (无载荷)。"""

    DATA_TYPE_INDEX = 0x0034
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> PingResponse:
        return cls()


# ---------------------------------------------------------------------------
# 7.3.2.62 超时时间更新请求 (0x003C, 16 bits / 2 bytes)
# ---------------------------------------------------------------------------

@dataclass
class TimeoutUpdateRequest:
    """超时时间更新请求。"""
    timeout: int  # 16 bits, 单位 10ms

    DATA_TYPE_INDEX = 0x003C
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return self.timeout.to_bytes(2, "big")

    @classmethod
    def unpack(cls, data: bytes) -> TimeoutUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:2], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.63 组播链路断开指示 (0x003D, 24 bits / 3 bytes)
# ---------------------------------------------------------------------------

@dataclass
class MulticastDisconnect:
    """组播链路断开指示。"""
    link_id: int       # 24 bits
    # 注: 标准定义 3 字节 (link_id 仅 24 bits, 无额外字段)

    DATA_TYPE_INDEX = 0x003D
    BYTE_LENGTH = 3

    def pack(self) -> bytes:
        return self.link_id.to_bytes(3, "big")

    @classmethod
    def unpack(cls, data: bytes) -> MulticastDisconnect:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(int.from_bytes(data[:3], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.35 链接态异步链路参数更新请求 (0x0020, 216 bits / 27 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncLinkParamRequest:
    """链接态异步链路参数更新请求。"""
    event_group_period_min: int   # 16 bits
    event_group_period_max: int   # 16 bits
    delay_period: int             # 16 bits
    timeout: int                  # 16 bits (10ms)
    expected_period_unit: int     # 8 bits
    effective_ref_slot: int       # 32 bits
    offsets: tuple[int, ...]      # 6 × 16 bits
    time_slot_length: int         # 8 bits
    time_slot_count: int          # 8 bits

    DATA_TYPE_INDEX = 0x0020
    BYTE_LENGTH = 27

    def pack(self) -> bytes:
        buf = struct.pack(">HHHH", self.event_group_period_min,
                          self.event_group_period_max,
                          self.delay_period, self.timeout)
        buf += bytes([self.expected_period_unit & 0xFF])
        buf += self.effective_ref_slot.to_bytes(4, "big")
        offs = (self.offsets + (0,) * 6)[:6]
        for o in offs:
            buf += struct.pack(">H", o & 0xFFFF)
        buf += bytes([self.time_slot_length & 0xFF, self.time_slot_count & 0xFF])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AsyncLinkParamRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        egpm, egpx, dp, to = struct.unpack(">HHHH", data[0:8])
        epu = data[8]
        ers = int.from_bytes(data[9:13], "big")
        offs = tuple(struct.unpack(">H", data[13 + i * 2:15 + i * 2])[0] for i in range(6))
        tsl = data[25]
        tsc = data[26]
        return cls(egpm, egpx, dp, to, epu, ers, offs, tsl, tsc)


# ---------------------------------------------------------------------------
# 7.3.2.36 链接态异步链路参数更新响应 (0x0021, 216 bits / 27 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncLinkParamResponse:
    """链接态异步链路参数更新响应 (与请求字段完全相同)。"""
    event_group_period_min: int
    event_group_period_max: int
    delay_period: int
    timeout: int
    expected_period_unit: int
    effective_ref_slot: int
    offsets: tuple[int, ...]
    time_slot_length: int
    time_slot_count: int

    DATA_TYPE_INDEX = 0x0021
    BYTE_LENGTH = 27

    def pack(self) -> bytes:
        buf = struct.pack(">HHHH", self.event_group_period_min,
                          self.event_group_period_max,
                          self.delay_period, self.timeout)
        buf += bytes([self.expected_period_unit & 0xFF])
        buf += self.effective_ref_slot.to_bytes(4, "big")
        offs = (self.offsets + (0,) * 6)[:6]
        for o in offs:
            buf += struct.pack(">H", o & 0xFFFF)
        buf += bytes([self.time_slot_length & 0xFF, self.time_slot_count & 0xFF])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AsyncLinkParamResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        egpm, egpx, dp, to = struct.unpack(">HHHH", data[0:8])
        epu = data[8]
        ers = int.from_bytes(data[9:13], "big")
        offs = tuple(struct.unpack(">H", data[13 + i * 2:15 + i * 2])[0] for i in range(6))
        tsl = data[25]
        tsc = data[26]
        return cls(egpm, egpx, dp, to, epu, ers, offs, tsl, tsc)


# ---------------------------------------------------------------------------
# 7.3.2.37 同步等时链路建链指示 (0x0022, 448 bits / 56 bytes)
# ---------------------------------------------------------------------------

@dataclass
class IsochronousLinkSetup:
    """同步等时链路建链指示。"""
    event_group_set_id: int       # 8
    event_group_id: int           # 8
    effective_slot: int           # 32
    event_group_period: int       # 16
    event_period: int             # 16
    intra_event_interval: int     # 16
    inter_event_interval: int     # 16
    event_count: int              # 8
    sync_anchor_delay: int        # 24
    sync_ref_delay: int           # 24
    scheduling_slot: int          # 3
    tx_rx_indication: int         # 1
    tx_adapt_mode: int            # 1
    rx_adapt_mode: int            # 1
    tx_link_id: int               # 24
    rx_link_id: int               # 24
    tx_frame_type: int            # 4
    rx_frame_type: int            # 4
    tx_bandwidth: int             # 2
    rx_bandwidth: int             # 2
    tx_pilot_density: int         # 2
    rx_pilot_density: int         # 2
    tx_sdu_max: int               # 12
    rx_sdu_max: int               # 12
    tx_sdu_period: int            # 20
    rx_sdu_period: int            # 20
    tx_pdu_max: int               # 11
    rx_pdu_max: int               # 11
    tx_max_time_offset: int       # 9
    rx_max_time_offset: int       # 9
    tx_new_pkt_count: int         # 4
    rx_new_pkt_count: int         # 4
    tx_crc_init: int              # 32
    rx_crc_init: int              # 32
    tx_discard_period: int        # 8
    rx_discard_period: int        # 8
    tx_crc_type: int              # 1
    rx_crc_type: int              # 1
    tx_feedback_type: int         # 6
    rx_feedback_type: int         # 3

    DATA_TYPE_INDEX = 0x0022
    BYTE_LENGTH = 56

    def pack(self) -> bytes:
        buf = bytes([self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])
        buf += self.effective_slot.to_bytes(4, "big")
        buf += struct.pack(">HHHHB", self.event_group_period, self.event_period,
                           self.intra_event_interval, self.inter_event_interval,
                           self.event_count)
        buf += self.sync_anchor_delay.to_bytes(3, "big")
        buf += self.sync_ref_delay.to_bytes(3, "big")
        # 编码剩余位域 (23 bytes = 184 bits)
        b = 0
        b = (b << 3) | (self.scheduling_slot & 0x07)
        b = (b << 2) | 0  # reserved
        b = (b << 1) | (self.tx_rx_indication & 0x01)
        b = (b << 1) | (self.tx_adapt_mode & 0x01)
        b = (b << 1) | (self.rx_adapt_mode & 0x01)
        b = (b << 24) | (self.tx_link_id & 0xFFFFFF)
        b = (b << 24) | (self.rx_link_id & 0xFFFFFF)
        b = (b << 4) | (self.tx_frame_type & 0x0F)
        b = (b << 4) | (self.rx_frame_type & 0x0F)
        b = (b << 2) | (self.tx_bandwidth & 0x03)
        b = (b << 2) | (self.rx_bandwidth & 0x03)
        b = (b << 2) | (self.tx_pilot_density & 0x03)
        b = (b << 2) | (self.rx_pilot_density & 0x03)
        b = (b << 12) | (self.tx_sdu_max & 0xFFF)
        b = (b << 12) | (self.rx_sdu_max & 0xFFF)
        b = (b << 20) | (self.tx_sdu_period & 0xFFFFF)
        b = (b << 20) | (self.rx_sdu_period & 0xFFFFF)
        b = (b << 11) | (self.tx_pdu_max & 0x7FF)
        b = (b << 11) | (self.rx_pdu_max & 0x7FF)
        b = (b << 9) | (self.tx_max_time_offset & 0x1FF)
        b = (b << 9) | (self.rx_max_time_offset & 0x1FF)
        b = (b << 4) | (self.tx_new_pkt_count & 0x0F)
        b = (b << 4) | (self.rx_new_pkt_count & 0x0F)
        buf += b.to_bytes(23, "big")
        buf += self.tx_crc_init.to_bytes(4, "big")
        buf += self.rx_crc_init.to_bytes(4, "big")
        buf += bytes([self.tx_discard_period & 0xFF, self.rx_discard_period & 0xFF])
        tail = ((self.tx_crc_type & 1) << 15
                | (self.rx_crc_type & 1) << 14
                | (self.tx_feedback_type & 0x3F) << 8
                | (self.rx_feedback_type & 0x07) << 5)
        buf += tail.to_bytes(2, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> IsochronousLinkSetup:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        egsi = data[0]
        egi = data[1]
        es = int.from_bytes(data[2:6], "big")
        egp, ep, iei, iei2, ec = struct.unpack(">HHHHB", data[6:15])
        sad = int.from_bytes(data[15:18], "big")
        srd = int.from_bytes(data[18:21], "big")
        b = int.from_bytes(data[21:44], "big")
        rnpc = b & 0x0F
        b >>= 4
        tnpc = b & 0x0F
        b >>= 4
        rmto = b & 0x1FF
        b >>= 9
        tmto = b & 0x1FF
        b >>= 9
        rpm = b & 0x7FF
        b >>= 11
        tpm = b & 0x7FF
        b >>= 11
        rsp = b & 0xFFFFF
        b >>= 20
        tsp = b & 0xFFFFF
        b >>= 20
        rsm = b & 0xFFF
        b >>= 12
        tsm = b & 0xFFF
        b >>= 12
        rpd = b & 0x03
        b >>= 2
        tpd = b & 0x03
        b >>= 2
        rb = b & 0x03
        b >>= 2
        tb = b & 0x03
        b >>= 2
        rft = b & 0x0F
        b >>= 4
        tft = b & 0x0F
        b >>= 4
        rli = b & 0xFFFFFF
        b >>= 24
        tli = b & 0xFFFFFF
        b >>= 24
        ram = b & 0x01
        b >>= 1
        tam = b & 0x01
        b >>= 1
        tri = b & 0x01
        b >>= 1
        b >>= 2  # reserved
        ss = b & 0x07
        tci = int.from_bytes(data[44:48], "big")
        rci = int.from_bytes(data[48:52], "big")
        tdp = data[52]
        rdp = data[53]
        tail = int.from_bytes(data[54:56], "big")
        tct = (tail >> 15) & 1
        rct = (tail >> 14) & 1
        tfb = (tail >> 8) & 0x3F
        rfb = (tail >> 5) & 0x07
        return cls(
            egsi, egi, es, egp, ep, iei, iei2, ec, sad, srd,
            ss, tri, tam, ram, tli, rli, tft, rft, tb, rb, tpd, rpd,
            tsm, rsm, tsp, rsp, tpm, rpm, tmto, rmto, tnpc, rnpc,
            tci, rci, tdp, rdp, tct, rct, tfb, rfb,
        )


# ---------------------------------------------------------------------------
# 7.3.2.38 同步等时链路参数交互请求 (0x0023, 416 bits / 52 bytes)
# ---------------------------------------------------------------------------

@dataclass
class IsochronousParamExchangeRequest:
    """同步等时链路参数交互请求。"""
    event_group_set_id: int
    event_group_id: int
    event_group_period: int
    event_period: int
    intra_event_interval: int
    inter_event_interval: int
    event_count: int
    sync_anchor_delay: int        # 24
    sync_ref_delay: int           # 24
    param_tag_id: int             # 4
    tx_rx_indication: int         # 1
    tx_adapt_mode: int            # 1
    rx_adapt_mode: int            # 1
    tx_link_id: int               # 24
    rx_link_id: int               # 24
    tx_frame_type: int            # 4
    rx_frame_type: int            # 4
    tx_bandwidth: int
    rx_bandwidth: int
    tx_pilot_density: int
    rx_pilot_density: int
    tx_sdu_max: int
    rx_sdu_max: int
    tx_sdu_period: int
    rx_sdu_period: int
    tx_pdu_max: int
    rx_pdu_max: int
    tx_max_time_offset: int
    rx_max_time_offset: int
    tx_new_pkt_count: int
    rx_new_pkt_count: int
    tx_crc_init: int
    rx_crc_init: int
    tx_discard_period: int
    rx_discard_period: int
    tx_crc_type: int
    rx_crc_type: int
    tx_feedback_type: int
    rx_feedback_type: int

    DATA_TYPE_INDEX = 0x0023
    BYTE_LENGTH = 52

    def pack(self) -> bytes:
        buf = bytes([self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])
        buf += struct.pack(">HHHHB", self.event_group_period, self.event_period,
                           self.intra_event_interval, self.inter_event_interval,
                           self.event_count)
        buf += self.sync_anchor_delay.to_bytes(3, "big")
        buf += self.sync_ref_delay.to_bytes(3, "big")
        b = 0
        b = (b << 4) | (self.param_tag_id & 0x0F)
        b = (b << 1) | 0  # reserved
        b = (b << 1) | (self.tx_rx_indication & 0x01)
        b = (b << 1) | (self.tx_adapt_mode & 0x01)
        b = (b << 1) | (self.rx_adapt_mode & 0x01)
        b = (b << 24) | (self.tx_link_id & 0xFFFFFF)
        b = (b << 24) | (self.rx_link_id & 0xFFFFFF)
        b = (b << 4) | (self.tx_frame_type & 0x0F)
        b = (b << 4) | (self.rx_frame_type & 0x0F)
        b = (b << 2) | (self.tx_bandwidth & 0x03)
        b = (b << 2) | (self.rx_bandwidth & 0x03)
        b = (b << 2) | (self.tx_pilot_density & 0x03)
        b = (b << 2) | (self.rx_pilot_density & 0x03)
        b = (b << 12) | (self.tx_sdu_max & 0xFFF)
        b = (b << 12) | (self.rx_sdu_max & 0xFFF)
        b = (b << 20) | (self.tx_sdu_period & 0xFFFFF)
        b = (b << 20) | (self.rx_sdu_period & 0xFFFFF)
        b = (b << 11) | (self.tx_pdu_max & 0x7FF)
        b = (b << 11) | (self.rx_pdu_max & 0x7FF)
        b = (b << 9) | (self.tx_max_time_offset & 0x1FF)
        b = (b << 9) | (self.rx_max_time_offset & 0x1FF)
        b = (b << 4) | (self.tx_new_pkt_count & 0x0F)
        b = (b << 4) | (self.rx_new_pkt_count & 0x0F)
        buf += b.to_bytes(23, "big")
        buf += self.tx_crc_init.to_bytes(4, "big")
        buf += self.rx_crc_init.to_bytes(4, "big")
        buf += bytes([self.tx_discard_period & 0xFF, self.rx_discard_period & 0xFF])
        tail = ((self.tx_crc_type & 1) << 15
                | (self.rx_crc_type & 1) << 14
                | (self.tx_feedback_type & 0x3F) << 8
                | (self.rx_feedback_type & 0x07) << 5)
        buf += tail.to_bytes(2, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> IsochronousParamExchangeRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        egsi = data[0]
        egi = data[1]
        egp, ep, iei, iei2, ec = struct.unpack(">HHHHB", data[2:11])
        sad = int.from_bytes(data[11:14], "big")
        srd = int.from_bytes(data[14:17], "big")
        b = int.from_bytes(data[17:40], "big")
        rnpc = b & 0x0F
        b >>= 4
        tnpc = b & 0x0F
        b >>= 4
        rmto = b & 0x1FF
        b >>= 9
        tmto = b & 0x1FF
        b >>= 9
        rpm = b & 0x7FF
        b >>= 11
        tpm = b & 0x7FF
        b >>= 11
        rsp = b & 0xFFFFF
        b >>= 20
        tsp = b & 0xFFFFF
        b >>= 20
        rsm = b & 0xFFF
        b >>= 12
        tsm = b & 0xFFF
        b >>= 12
        rpd = b & 0x03
        b >>= 2
        tpd = b & 0x03
        b >>= 2
        rb = b & 0x03
        b >>= 2
        tb = b & 0x03
        b >>= 2
        rft = b & 0x0F
        b >>= 4
        tft = b & 0x0F
        b >>= 4
        rli = b & 0xFFFFFF
        b >>= 24
        tli = b & 0xFFFFFF
        b >>= 24
        ram = b & 0x01
        b >>= 1
        tam = b & 0x01
        b >>= 1
        tri = b & 0x01
        b >>= 1
        b >>= 1  # reserved
        pti = b & 0x0F
        tci = int.from_bytes(data[40:44], "big")
        rci = int.from_bytes(data[44:48], "big")
        tdp = data[48]
        rdp = data[49]
        tail = int.from_bytes(data[50:52], "big")
        tct = (tail >> 15) & 1
        rct = (tail >> 14) & 1
        tfb = (tail >> 8) & 0x3F
        rfb = (tail >> 5) & 0x07
        return cls(
            egsi, egi, egp, ep, iei, iei2, ec, sad, srd,
            pti, tri, tam, ram, tli, rli, tft, rft, tb, rb, tpd, rpd,
            tsm, rsm, tsp, rsp, tpm, rpm, tmto, rmto, tnpc, rnpc,
            tci, rci, tdp, rdp, tct, rct, tfb, rfb,
        )


# ---------------------------------------------------------------------------
# 7.3.2.39 同步等时链路参数交互响应 (0x0024, 416 bits / 52 bytes)
# ---------------------------------------------------------------------------

@dataclass
class IsochronousParamExchangeResponse(IsochronousParamExchangeRequest):
    """同步等时链路参数交互响应 (与请求字段相同)。"""

    DATA_TYPE_INDEX = 0x0024
    BYTE_LENGTH = 52

    @classmethod
    def unpack(cls, data: bytes) -> IsochronousParamExchangeResponse:
        req = IsochronousParamExchangeRequest.unpack(data)
        return cls(**{f.name: getattr(req, f.name) for f in fields(req)})


# ---------------------------------------------------------------------------
# 7.3.2.40 同步等时链路参数更新请求 (0x0025, 24 bits / 3 bytes)
# ---------------------------------------------------------------------------

@dataclass
class IsochronousParamUpdateRequest:
    """同步等时链路参数更新请求。"""
    param_tag_id: int        # 4 bits
    event_group_set_id: int  # 8 bits
    event_group_id: int      # 8 bits

    DATA_TYPE_INDEX = 0x0025
    BYTE_LENGTH = 3

    def pack(self) -> bytes:
        b0 = (self.param_tag_id & 0x0F) << 4
        return bytes([b0, self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> IsochronousParamUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls((data[0] >> 4) & 0x0F, data[1], data[2])


# ---------------------------------------------------------------------------
# 7.3.2.41 同步等时链路参数更新指示 (0x0026, 72 bits / 9 bytes)
# ---------------------------------------------------------------------------

@dataclass
class IsochronousParamUpdateIndication:
    """同步等时链路参数更新指示。"""
    param_tag_id: int            # 4 bits
    event_group_set_id: int      # 8 bits
    event_group_id: int          # 8 bits
    effective_ref_slot: int      # 32 bits
    event_group_offset: int      # 16 bits

    DATA_TYPE_INDEX = 0x0026
    BYTE_LENGTH = 9

    def pack(self) -> bytes:
        b0 = (self.param_tag_id & 0x0F) << 4
        buf = bytes([b0, self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])
        buf += self.effective_ref_slot.to_bytes(4, "big")
        buf += struct.pack(">H", self.event_group_offset)
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> IsochronousParamUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        pti = (data[0] >> 4) & 0x0F
        egsi = data[1]
        egi = data[2]
        ers = int.from_bytes(data[3:7], "big")
        ego = struct.unpack(">H", data[7:9])[0]
        return cls(pti, egsi, egi, ers, ego)


# ---------------------------------------------------------------------------
# 7.3.2.42 链接态广播链路建立指示 (0x0027, 可变长)
#   固定部分 + 80 bit 跳频地图 (2.4GHz) = 68 bytes
# ---------------------------------------------------------------------------

@dataclass
class BroadcastLinkSetup:
    """链接态广播链路建立指示 (2.4GHz 80-bit 跳频地图版本)。"""
    transmission_type: int        # 1 bit
    adapt_mode: int               # 1 bit
    event_group_set_id: int       # 8
    event_group_count: int        # 8
    event_group_id: int           # 8
    effective_slot: int           # 32
    event_group_interval: int     # 8
    event_group_period: int       # 16
    event_period: int             # 16
    event_count: int              # 8
    base_link_id: int             # 24
    frame_type: int               # 4
    bandwidth: int                # 2
    pilot_density: int            # 2
    sdu_max: int                  # 12
    sdu_period: int               # 20
    pdu_max: int                  # 11
    new_pkt_count: int            # 4
    crc_type: int                 # 1
    crc_base_init: int            # 32
    hop_map: bytes                # 80 bits = 10 bytes
    sync_anchor_delay: int        # 24
    sync_ref_delay: int           # 24

    DATA_TYPE_INDEX = 0x0027
    BYTE_LENGTH = 44  # 不含可选加密字段

    def pack(self) -> bytes:
        b0 = ((self.transmission_type & 1) << 7
              | (self.adapt_mode & 1) << 6)
        buf = bytes([b0, self.event_group_set_id & 0xFF,
                     self.event_group_count & 0xFF, self.event_group_id & 0xFF])
        buf += self.effective_slot.to_bytes(4, "big")
        buf += bytes([self.event_group_interval & 0xFF])
        buf += struct.pack(">HHB", self.event_group_period,
                           self.event_period, self.event_count)
        buf += self.base_link_id.to_bytes(3, "big")
        # 位域: frame_type(4) + bandwidth(2) + pilot_density(2) = 8 bits
        b_cfg = ((self.frame_type & 0x0F) << 4
                 | (self.bandwidth & 0x03) << 2
                 | (self.pilot_density & 0x03))
        buf += bytes([b_cfg])
        # sdu_max(12) + sdu_period(20) = 32 bits
        sp = ((self.sdu_max & 0xFFF) << 20) | (self.sdu_period & 0xFFFFF)
        buf += sp.to_bytes(4, "big")
        # pdu_max(11) + new_pkt_count(4) + crc_type(1) = 16 bits
        pc = ((self.pdu_max & 0x7FF) << 5
              | (self.new_pkt_count & 0x0F) << 1
              | (self.crc_type & 1))
        buf += pc.to_bytes(2, "big")
        buf += self.crc_base_init.to_bytes(4, "big")
        hm = self.hop_map[:10] if len(self.hop_map) >= 10 else self.hop_map + b"\x00" * (10 - len(self.hop_map))
        buf += hm
        buf += self.sync_anchor_delay.to_bytes(3, "big")
        buf += self.sync_ref_delay.to_bytes(3, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> BroadcastLinkSetup:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        tt = (data[0] >> 7) & 1
        am = (data[0] >> 6) & 1
        egsi = data[1]
        egc = data[2]
        egi = data[3]
        es = int.from_bytes(data[4:8], "big")
        egi2 = data[8]
        egp, ep, ec = struct.unpack(">HHB", data[9:14])
        bli = int.from_bytes(data[14:17], "big")
        bcfg = data[17]
        ft = (bcfg >> 4) & 0x0F
        bw = (bcfg >> 2) & 0x03
        pd = bcfg & 0x03
        sp = int.from_bytes(data[18:22], "big")
        sm = (sp >> 20) & 0xFFF
        spr = sp & 0xFFFFF
        pc = int.from_bytes(data[22:24], "big")
        pm = (pc >> 5) & 0x7FF
        npc = (pc >> 1) & 0x0F
        ct = pc & 1
        cbi = int.from_bytes(data[24:28], "big")
        hm = data[28:38]
        sad = int.from_bytes(data[38:41], "big")
        srd = int.from_bytes(data[41:44], "big") if len(data) >= 44 else 0
        return cls(tt, am, egsi, egc, egi, es, egi2, egp, ep, ec,
                   bli, ft, bw, pd, sm, spr, pm, npc, ct, cbi, hm, sad, srd)


# ---------------------------------------------------------------------------
# 7.3.2.43 广播链路参数更新指示 (0x0028, 240 bits / 30 bytes)
# ---------------------------------------------------------------------------

@dataclass
class BroadcastLinkParamUpdate:
    """广播链路参数更新指示。"""
    event_group_set_id: int
    event_group_count: int
    event_group_id: int
    event_group_period: int       # 16
    event_period: int             # 16
    event_count: int              # 8
    frame_type: int               # 4
    bandwidth: int                # 2
    pilot_density: int            # 2
    sdu_max: int                  # 12
    new_pkt_count: int            # 4
    adapt_mode: int               # 1
    sdu_period: int               # 20
    pdu_max: int                  # 11
    crc_type: int                 # 1
    crc_base_init: int            # 32
    sync_anchor_delay: int        # 24
    sync_ref_delay: int           # 24
    effective_ref_slot: int       # 32
    event_group_offset: int       # 16

    DATA_TYPE_INDEX = 0x0028
    BYTE_LENGTH = 32

    def pack(self) -> bytes:
        buf = bytes([self.event_group_set_id & 0xFF,
                     self.event_group_count & 0xFF,
                     self.event_group_id & 0xFF])
        buf += struct.pack(">HHB", self.event_group_period,
                           self.event_period, self.event_count)
        b_cfg = ((self.frame_type & 0x0F) << 4
                 | (self.bandwidth & 0x03) << 2
                 | (self.pilot_density & 0x03))
        buf += bytes([b_cfg])
        # sdu_max(12) + new_pkt_count(4) + adapt_mode(1) + reserved(7) = 24 bits
        sm_np = ((self.sdu_max & 0xFFF) << 12
                 | (self.new_pkt_count & 0x0F) << 8
                 | (self.adapt_mode & 1) << 7)
        buf += sm_np.to_bytes(3, "big")
        # sdu_period(20) + pdu_max(11) + crc_type(1) = 32 bits
        sp_pm = ((self.sdu_period & 0xFFFFF) << 12
                 | (self.pdu_max & 0x7FF) << 1
                 | (self.crc_type & 1))
        buf += sp_pm.to_bytes(4, "big")
        buf += self.crc_base_init.to_bytes(4, "big")
        buf += self.sync_anchor_delay.to_bytes(3, "big")
        buf += self.sync_ref_delay.to_bytes(3, "big")
        buf += self.effective_ref_slot.to_bytes(4, "big")
        buf += struct.pack(">H", self.event_group_offset)
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> BroadcastLinkParamUpdate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        egsi, egc, egi = data[0], data[1], data[2]
        egp, ep, ec = struct.unpack(">HHB", data[3:8])
        bcfg = data[8]
        ft = (bcfg >> 4) & 0x0F
        bw = (bcfg >> 2) & 0x03
        pd = bcfg & 0x03
        sm_np = int.from_bytes(data[9:12], "big")
        sm = (sm_np >> 12) & 0xFFF
        npc = (sm_np >> 8) & 0x0F
        am = (sm_np >> 7) & 1
        sp_pm = int.from_bytes(data[12:16], "big")
        spr = (sp_pm >> 12) & 0xFFFFF
        pm = (sp_pm >> 1) & 0x7FF
        ct = sp_pm & 1
        cbi = int.from_bytes(data[16:20], "big")
        sad = int.from_bytes(data[20:23], "big")
        srd = int.from_bytes(data[23:26], "big")
        ers = int.from_bytes(data[26:30], "big")
        ego = 0
        if len(data) >= 32:
            ego = struct.unpack(">H", data[30:32])[0]
        return cls(egsi, egc, egi, egp, ep, ec, ft, bw, pd,
                   sm, npc, am, spr, pm, ct, cbi, sad, srd, ers, ego)


# ---------------------------------------------------------------------------
# 7.3.2.44 广播链路跳频地图更新指示 (0x0029, 112 bits / 14 bytes)
# ---------------------------------------------------------------------------

@dataclass
class BroadcastHopMapUpdate:
    """广播链路跳频地图更新指示。"""
    hop_map: bytes               # 80 bits = 10 bytes
    effective_slot: int          # 32 bits

    DATA_TYPE_INDEX = 0x0029
    BYTE_LENGTH = 14

    def pack(self) -> bytes:
        hm = self.hop_map[:10] if len(self.hop_map) >= 10 else self.hop_map + b"\x00" * (10 - len(self.hop_map))
        return hm + self.effective_slot.to_bytes(4, "big")

    @classmethod
    def unpack(cls, data: bytes) -> BroadcastHopMapUpdate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[:10], int.from_bytes(data[10:14], "big"))


# ---------------------------------------------------------------------------
# 7.3.2.46 系统管理帧参数更新请求 (0x002B, 64 bits / 8 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SMFParamUpdateRequest:
    """系统管理帧参数更新请求。"""
    smf_period: int              # 16 bits
    smf_start_offset: int        # 8 bits
    link_id: int                 # 24 bits
    frame_type: int              # 4 bits
    bandwidth: int               # 2 bits
    pilot_density: int           # 2 bits
    crc_type: int                # 1 bit

    DATA_TYPE_INDEX = 0x002B
    BYTE_LENGTH = 8

    def pack(self) -> bytes:
        buf = struct.pack(">HB", self.smf_period, self.smf_start_offset)
        buf += self.link_id.to_bytes(3, "big")
        b_cfg = ((self.frame_type & 0x0F) << 4
                 | (self.bandwidth & 0x03) << 2
                 | (self.pilot_density & 0x03))
        b_tail = ((self.crc_type & 1) << 7)
        buf += bytes([b_cfg, b_tail])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> SMFParamUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        sp = struct.unpack(">H", data[0:2])[0]
        so = data[2]
        li = int.from_bytes(data[3:6], "big")
        ft = (data[6] >> 4) & 0x0F
        bw = (data[6] >> 2) & 0x03
        pd = data[6] & 0x03
        ct = (data[7] >> 7) & 1
        return cls(sp, so, li, ft, bw, pd, ct)


# ---------------------------------------------------------------------------
# 7.3.2.47 系统管理帧参数更新指示 (0x002C, 96 bits / 12 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SMFParamUpdateIndication:
    """系统管理帧参数更新指示。"""
    smf_period: int
    smf_start_offset: int
    link_id: int
    frame_type: int
    bandwidth: int
    pilot_density: int
    crc_type: int
    crc_init: int                # 32 bits -> 但标准说 96 bits total, 所以这里不含

    DATA_TYPE_INDEX = 0x002C
    BYTE_LENGTH = 12

    def pack(self) -> bytes:
        buf = struct.pack(">HB", self.smf_period, self.smf_start_offset)
        buf += self.link_id.to_bytes(3, "big")
        b_cfg = ((self.frame_type & 0x0F) << 4
                 | (self.bandwidth & 0x03) << 2
                 | (self.pilot_density & 0x03))
        b_tail = ((self.crc_type & 1) << 7)
        buf += bytes([b_cfg, b_tail])
        buf += self.crc_init.to_bytes(4, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> SMFParamUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        sp = struct.unpack(">H", data[0:2])[0]
        so = data[2]
        li = int.from_bytes(data[3:6], "big")
        ft = (data[6] >> 4) & 0x0F
        bw = (data[6] >> 2) & 0x03
        pd = data[6] & 0x03
        ct = (data[7] >> 7) & 1
        ci = int.from_bytes(data[8:12], "big")
        return cls(sp, so, li, ft, bw, pd, ct, ci)


# ---------------------------------------------------------------------------
# 7.3.2.48 系统管理帧时间片更新请求 (0x002D, 104 bits / 13 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SMFTimeSlotUpdateRequest:
    """系统管理帧时间片更新请求。"""
    link_id: int              # 24 bits
    current_offset: int       # 16 bits
    offsets: tuple[int, ...]  # 4 × 16 bits

    DATA_TYPE_INDEX = 0x002D
    BYTE_LENGTH = 13

    def pack(self) -> bytes:
        buf = self.link_id.to_bytes(3, "big")
        buf += struct.pack(">H", self.current_offset)
        offs = (self.offsets + (0,) * 4)[:4]
        for o in offs:
            buf += struct.pack(">H", o & 0xFFFF)
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> SMFTimeSlotUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        li = int.from_bytes(data[0:3], "big")
        co = struct.unpack(">H", data[3:5])[0]
        offs = tuple(struct.unpack(">H", data[5 + i * 2:7 + i * 2])[0] for i in range(4))
        return cls(li, co, offs)


# ---------------------------------------------------------------------------
# 7.3.2.49 系统管理帧时间片更新响应 (0x002E, 72 bits / 9 bytes)
# ---------------------------------------------------------------------------

@dataclass
class SMFTimeSlotUpdateResponse:
    """系统管理帧时间片更新响应。"""
    link_id: int              # 24 bits
    offset: int               # 16 bits
    effective_slot: int       # 32 bits

    DATA_TYPE_INDEX = 0x002E
    BYTE_LENGTH = 9

    def pack(self) -> bytes:
        buf = self.link_id.to_bytes(3, "big")
        buf += struct.pack(">H", self.offset)
        buf += self.effective_slot.to_bytes(4, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> SMFTimeSlotUpdateResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        li = int.from_bytes(data[0:3], "big")
        o = struct.unpack(">H", data[3:5])[0]
        es = int.from_bytes(data[5:9], "big")
        return cls(li, o, es)


# ===================================================================
# 0x0035 - 0x003B  5GHz / 多级收发间隔
# ===================================================================

# ---------------------------------------------------------------------------
# 7.3.2.50 5GHz 频段信道状态指示 (0x0035, 400 bits / 50 bytes)
# ---------------------------------------------------------------------------

@dataclass
class Channel5GStatusIndication:
    """5GHz 频段信道状态指示 (400-bit 信道分类)。"""
    channel_classification: bytes  # 50 bytes

    DATA_TYPE_INDEX = 0x0035
    BYTE_LENGTH = 50

    def pack(self) -> bytes:
        return (self.channel_classification + b"\x00" * 50)[:50]

    @classmethod
    def unpack(cls, data: bytes) -> Channel5GStatusIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[:50])


# ---------------------------------------------------------------------------
# 5GHz 跳频地图更新指示 (0x0036 数据链路 / 0x003B 广播链路, 232 bits / 29 bytes)
# ---------------------------------------------------------------------------

@dataclass
class HopMap5GUpdate:
    """5GHz 跳频地图更新指示 (数据链路)。"""
    hop_map: bytes          # 200 bits = 25 bytes
    effective_slot: int     # 32 bits

    DATA_TYPE_INDEX = 0x0036
    BYTE_LENGTH = 29

    def pack(self) -> bytes:
        hm = (self.hop_map + b"\x00" * 25)[:25]
        return hm + self.effective_slot.to_bytes(4, "big")

    @classmethod
    def unpack(cls, data: bytes) -> HopMap5GUpdate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[:25], int.from_bytes(data[25:29], "big"))


@dataclass
class BroadcastHopMap5GUpdate(HopMap5GUpdate):
    """5GHz 广播链路跳频地图更新指示。"""

    DATA_TYPE_INDEX = 0x003B
    BYTE_LENGTH = 29

    @classmethod
    def unpack(cls, data: bytes) -> BroadcastHopMap5GUpdate:
        base = HopMap5GUpdate.unpack(data)
        return cls(base.hop_map, base.effective_slot)


# ---------------------------------------------------------------------------
# 多级收发间隔更新 (0x0037 请求 / 0x0038 响应, 248 bits / 31 bytes)
# ---------------------------------------------------------------------------

@dataclass
class MultiIntervalUpdateRequest:
    """多级收发间隔更新请求 (31 个间隔字节)。"""
    intervals: bytes  # 31 bytes

    DATA_TYPE_INDEX = 0x0037
    BYTE_LENGTH = 31

    def pack(self) -> bytes:
        return (self.intervals + b"\x00" * 31)[:31]

    @classmethod
    def unpack(cls, data: bytes) -> MultiIntervalUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[:31])


@dataclass
class MultiIntervalUpdateResponse(MultiIntervalUpdateRequest):
    """多级收发间隔更新响应 (与请求结构相同)。"""

    DATA_TYPE_INDEX = 0x0038
    BYTE_LENGTH = 31

    @classmethod
    def unpack(cls, data: bytes) -> MultiIntervalUpdateResponse:
        base = MultiIntervalUpdateRequest.unpack(data)
        return cls(base.intervals)


# ---------------------------------------------------------------------------
# 多级收发间隔更新指示 (0x0039, 288 bits / 36 bytes)
# ---------------------------------------------------------------------------

@dataclass
class MultiIntervalUpdateIndication:
    """多级收发间隔更新指示。"""
    intervals: bytes              # 31 bytes
    update_flags: int             # 6 bits (打包在一个字节的高 6 位)
    effective_slot: int           # 32 bits

    DATA_TYPE_INDEX = 0x0039
    BYTE_LENGTH = 36

    def pack(self) -> bytes:
        buf = (self.intervals + b"\x00" * 31)[:31]
        buf += bytes([(self.update_flags & 0x3F) << 2])
        buf += self.effective_slot.to_bytes(4, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> MultiIntervalUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        intervals = data[:31]
        uf = (data[31] >> 2) & 0x3F
        es = int.from_bytes(data[32:36], "big")
        return cls(intervals, uf, es)


# ===================================================================
# 0x003E - 0x0043  系统时间 / 异步组播
# ===================================================================

# ---------------------------------------------------------------------------
# 系统时间指示 (0x003E, 可变长)
# ---------------------------------------------------------------------------

@dataclass
class SystemTimeIndication:
    """系统时间指示 (可变长, 按原始载荷存储)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x003E
    BYTE_LENGTH = 0  # 可变

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> SystemTimeIndication:
        return cls(bytes(data))


# ---------------------------------------------------------------------------
# 异步组播链路建链指示 (0x003F, 360 bits / 45 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncMulticastLinkSetup:
    """异步组播链路建链指示。"""
    event_group_set_id: int       # 8
    event_group_id: int           # 8
    effective_slot: int           # 32
    event_group_period: int       # 16
    event_period: int             # 16
    intra_event_interval: int     # 16
    inter_event_interval: int     # 16
    scheduling_slot: int          # 3
    tx_rx_indication: int         # 1
    tx_link_id: int               # 24
    rx_link_id: int               # 24
    tx_frame_type: int            # 4
    rx_frame_type: int            # 4
    tx_bandwidth: int             # 2
    rx_bandwidth: int             # 2
    tx_pilot_density: int         # 2
    rx_pilot_density: int         # 2
    tx_sdu_max: int               # 12
    rx_sdu_max: int               # 12
    tx_sdu_period: int            # 20
    rx_sdu_period: int            # 20
    tx_pdu_max: int               # 11
    rx_pdu_max: int               # 11
    tx_max_time_offset: int       # 9
    rx_max_time_offset: int       # 9
    tx_crc_init: int              # 32
    rx_crc_init: int              # 32
    tx_crc_type: int              # 1
    rx_crc_type: int              # 1
    tx_feedback_type: int         # 6
    rx_feedback_type: int         # 3

    DATA_TYPE_INDEX = 0x003F
    BYTE_LENGTH = 45

    def pack(self) -> bytes:
        buf = bytes([self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])
        buf += self.effective_slot.to_bytes(4, "big")
        buf += struct.pack(">HHHH", self.event_group_period, self.event_period,
                           self.intra_event_interval, self.inter_event_interval)
        # 剩余字段合并为单一位域: 247 bits + 1 reserved = 248 bits = 31 bytes
        b = 0
        b = (b << 3) | (self.scheduling_slot & 0x07)
        b = (b << 1) | (self.tx_rx_indication & 0x01)
        b = (b << 24) | (self.tx_link_id & 0xFFFFFF)
        b = (b << 24) | (self.rx_link_id & 0xFFFFFF)
        b = (b << 4) | (self.tx_frame_type & 0x0F)
        b = (b << 4) | (self.rx_frame_type & 0x0F)
        b = (b << 2) | (self.tx_bandwidth & 0x03)
        b = (b << 2) | (self.rx_bandwidth & 0x03)
        b = (b << 2) | (self.tx_pilot_density & 0x03)
        b = (b << 2) | (self.rx_pilot_density & 0x03)
        b = (b << 12) | (self.tx_sdu_max & 0xFFF)
        b = (b << 12) | (self.rx_sdu_max & 0xFFF)
        b = (b << 20) | (self.tx_sdu_period & 0xFFFFF)
        b = (b << 20) | (self.rx_sdu_period & 0xFFFFF)
        b = (b << 11) | (self.tx_pdu_max & 0x7FF)
        b = (b << 11) | (self.rx_pdu_max & 0x7FF)
        b = (b << 9) | (self.tx_max_time_offset & 0x1FF)
        b = (b << 9) | (self.rx_max_time_offset & 0x1FF)
        b = (b << 32) | (self.tx_crc_init & 0xFFFFFFFF)
        b = (b << 32) | (self.rx_crc_init & 0xFFFFFFFF)
        b = (b << 1) | (self.tx_crc_type & 1)
        b = (b << 1) | (self.rx_crc_type & 1)
        b = (b << 6) | (self.tx_feedback_type & 0x3F)
        b = (b << 3) | (self.rx_feedback_type & 0x07)
        b <<= 1  # 1 bit reserved
        buf += b.to_bytes(31, "big")
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AsyncMulticastLinkSetup:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        egsi = data[0]
        egi = data[1]
        es = int.from_bytes(data[2:6], "big")
        egp, ep, iei, iei2 = struct.unpack(">HHHH", data[6:14])
        b = int.from_bytes(data[14:45], "big")
        b >>= 1  # reserved
        rfb = b & 0x07
        b >>= 3
        tfb = b & 0x3F
        b >>= 6
        rct = b & 0x01
        b >>= 1
        tct = b & 0x01
        b >>= 1
        rci = b & 0xFFFFFFFF
        b >>= 32
        tci = b & 0xFFFFFFFF
        b >>= 32
        rmto = b & 0x1FF
        b >>= 9
        tmto = b & 0x1FF
        b >>= 9
        rpm = b & 0x7FF
        b >>= 11
        tpm = b & 0x7FF
        b >>= 11
        rsp = b & 0xFFFFF
        b >>= 20
        tsp = b & 0xFFFFF
        b >>= 20
        rsm = b & 0xFFF
        b >>= 12
        tsm = b & 0xFFF
        b >>= 12
        rpd = b & 0x03
        b >>= 2
        tpd = b & 0x03
        b >>= 2
        rb = b & 0x03
        b >>= 2
        tb = b & 0x03
        b >>= 2
        rft = b & 0x0F
        b >>= 4
        tft = b & 0x0F
        b >>= 4
        rli = b & 0xFFFFFF
        b >>= 24
        tli = b & 0xFFFFFF
        b >>= 24
        tri = b & 0x01
        b >>= 1
        ss = b & 0x07
        return cls(
            egsi, egi, es, egp, ep, iei, iei2,
            ss, tri, tli, rli, tft, rft, tb, rb, tpd, rpd,
            tsm, rsm, tsp, rsp, tpm, rpm, tmto, rmto,
            tci, rci, tct, rct, tfb, rfb,
        )


# ---------------------------------------------------------------------------
# 异步组播链路参数交互请求/响应 (0x0040/0x0041, 328 bits / 41 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncMulticastParamExchangeRequest:
    """异步组播链路参数交互请求。"""
    payload: bytes  # 41 bytes (复杂位域, 按原始载荷存储)

    DATA_TYPE_INDEX = 0x0040
    BYTE_LENGTH = 41

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 41)[:41]

    @classmethod
    def unpack(cls, data: bytes) -> AsyncMulticastParamExchangeRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:41]))


@dataclass
class AsyncMulticastParamExchangeResponse:
    """异步组播链路参数交互响应 (与请求结构相同)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0041
    BYTE_LENGTH = 41

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 41)[:41]

    @classmethod
    def unpack(cls, data: bytes) -> AsyncMulticastParamExchangeResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:41]))


# ---------------------------------------------------------------------------
# 异步组播链路参数更新请求 (0x0042, 24 bits / 3 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncMulticastParamUpdateRequest:
    """异步组播链路参数更新请求。"""
    param_tag_id: int            # 4 bits
    event_group_set_id: int      # 8
    event_group_id: int          # 8

    DATA_TYPE_INDEX = 0x0042
    BYTE_LENGTH = 3

    def pack(self) -> bytes:
        b0 = (self.param_tag_id & 0x0F) << 4
        return bytes([b0, self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> AsyncMulticastParamUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls((data[0] >> 4) & 0x0F, data[1], data[2])


# ---------------------------------------------------------------------------
# 异步组播链路参数更新指示 (0x0043, 72 bits / 9 bytes)
# ---------------------------------------------------------------------------

@dataclass
class AsyncMulticastParamUpdateIndication:
    """异步组播链路参数更新指示。"""
    param_tag_id: int
    event_group_set_id: int
    event_group_id: int
    effective_ref_slot: int      # 32
    event_group_offset: int      # 16

    DATA_TYPE_INDEX = 0x0043
    BYTE_LENGTH = 9

    def pack(self) -> bytes:
        b0 = (self.param_tag_id & 0x0F) << 4
        buf = bytes([b0, self.event_group_set_id & 0xFF, self.event_group_id & 0xFF])
        buf += self.effective_ref_slot.to_bytes(4, "big")
        buf += struct.pack(">H", self.event_group_offset)
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> AsyncMulticastParamUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(
            (data[0] >> 4) & 0x0F, data[1], data[2],
            int.from_bytes(data[3:7], "big"),
            struct.unpack(">H", data[7:9])[0],
        )


# ===================================================================
# 0x0044 - 0x0050  窄带跳频测量
# ===================================================================

# ---------------------------------------------------------------------------
# 零载荷信令基类
# ---------------------------------------------------------------------------

@dataclass
class _ZeroPayload:
    """零载荷信令基类。"""

    def pack(self) -> bytes:
        return b""

    @classmethod
    def unpack(cls, data: bytes):
        return cls()


@dataclass
class NarrowbandMeasCapRequest(_ZeroPayload):
    """窄带跳频测量能力请求。"""
    DATA_TYPE_INDEX = 0x0044
    BYTE_LENGTH = 0


# ---------------------------------------------------------------------------
# 窄带跳频测量能力响应 (0x0045, 256 bits / 32 bytes)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandMeasCapResponse:
    """窄带跳频测量能力响应。"""
    payload: bytes  # 32 bytes

    DATA_TYPE_INDEX = 0x0045
    BYTE_LENGTH = 32

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 32)[:32]

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasCapResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:32]))


# ---------------------------------------------------------------------------
# 窄带跳频测量频点表配置更新指示
# 0x0046 (2.4GHz, 11B), 0x0047 (5.1GHz, 26B), 0x0048 (5.8GHz, 17B)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandFreqTable24Update:
    """窄带跳频测量频点表配置更新指示 (2.4GHz)。"""
    config_index: int    # 8 bits
    freq_table: bytes    # 80 bits = 10 bytes

    DATA_TYPE_INDEX = 0x0046
    BYTE_LENGTH = 11

    def pack(self) -> bytes:
        ft = (self.freq_table + b"\x00" * 10)[:10]
        return bytes([self.config_index & 0xFF]) + ft

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandFreqTable24Update:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1:11])


@dataclass
class NarrowbandFreqTable51Update:
    """窄带跳频测量频点表配置更新指示 (5.1GHz)。"""
    config_index: int
    freq_table: bytes    # 200 bits = 25 bytes

    DATA_TYPE_INDEX = 0x0047
    BYTE_LENGTH = 26

    def pack(self) -> bytes:
        ft = (self.freq_table + b"\x00" * 25)[:25]
        return bytes([self.config_index & 0xFF]) + ft

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandFreqTable51Update:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1:26])


@dataclass
class NarrowbandFreqTable58Update:
    """窄带跳频测量频点表配置更新指示 (5.8GHz)。"""
    config_index: int
    freq_table: bytes    # 128 bits = 16 bytes

    DATA_TYPE_INDEX = 0x0048
    BYTE_LENGTH = 17

    def pack(self) -> bytes:
        ft = (self.freq_table + b"\x00" * 16)[:16]
        return bytes([self.config_index & 0xFF]) + ft

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandFreqTable58Update:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1:17])


# ---------------------------------------------------------------------------
# 窄带跳频测量信号配置 (0x0049, 可变长)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandMeasConfig:
    """窄带跳频测量信号配置 (可变长, 按原始载荷存储)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0049
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasConfig:
        return cls(bytes(data))


# ---------------------------------------------------------------------------
# 窄带跳频测量信息上报 (0x004A, 可变长)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandMeasReport:
    """窄带跳频测量信息上报 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x004A
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasReport:
        return cls(bytes(data))


# ---------------------------------------------------------------------------
# 窄带跳频测量行为指示 (0x004B, 48 bits / 6 bytes)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandMeasAction:
    """窄带跳频测量行为指示。"""
    config_index: int        # 8
    start_slot: int          # 32
    action_config: int       # 8

    DATA_TYPE_INDEX = 0x004B
    BYTE_LENGTH = 6

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF]) + self.start_slot.to_bytes(4, "big") + bytes([self.action_config & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasAction:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], int.from_bytes(data[1:5], "big"), data[5])


# ---------------------------------------------------------------------------
# 坐标信息请求 (0x004C, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class CoordinateRequest(_ZeroPayload):
    """坐标信息请求。"""
    DATA_TYPE_INDEX = 0x004C
    BYTE_LENGTH = 0


# ---------------------------------------------------------------------------
# 坐标信息上报/配置 (0x004D/0x004E, 192 bits / 24 bytes)
# ---------------------------------------------------------------------------

@dataclass
class CoordinateReport:
    """坐标信息上报。"""
    rel_x: int    # 32
    rel_y: int    # 32
    rel_z: int    # 32
    abs_lon: int  # 32 (经度)
    abs_lat: int  # 32 (纬度)
    abs_alt: int  # 32 (海拔)

    DATA_TYPE_INDEX = 0x004D
    BYTE_LENGTH = 24

    def pack(self) -> bytes:
        return struct.pack(">iiiiii", self.rel_x, self.rel_y, self.rel_z,
                           self.abs_lon, self.abs_lat, self.abs_alt)

    @classmethod
    def unpack(cls, data: bytes) -> CoordinateReport:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        vals = struct.unpack(">iiiiii", data[:24])
        return cls(*vals)


@dataclass
class CoordinateConfig:
    """坐标信息配置 (与上报结构相同)。"""
    rel_x: int
    rel_y: int
    rel_z: int
    abs_lon: int
    abs_lat: int
    abs_alt: int

    DATA_TYPE_INDEX = 0x004E
    BYTE_LENGTH = 24

    def pack(self) -> bytes:
        return struct.pack(">iiiiii", self.rel_x, self.rel_y, self.rel_z,
                           self.abs_lon, self.abs_lat, self.abs_alt)

    @classmethod
    def unpack(cls, data: bytes) -> CoordinateConfig:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        vals = struct.unpack(">iiiiii", data[:24])
        return cls(*vals)


# ---------------------------------------------------------------------------
# 窄带跳频测量时延信息请求 (0x004F, 0 bytes)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandDelayRequest(_ZeroPayload):
    """窄带跳频测量时延信息请求。"""
    DATA_TYPE_INDEX = 0x004F
    BYTE_LENGTH = 0


# ---------------------------------------------------------------------------
# 窄带跳频测量时延信息响应 (0x0050, 可变长)
# ---------------------------------------------------------------------------

@dataclass
class NarrowbandDelayResponse:
    """窄带跳频测量时延信息响应 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0050
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandDelayResponse:
        return cls(bytes(data))


# ===================================================================
# 0x0051  异步 TT 链路建链指示 (可变长)
# ===================================================================

@dataclass
class AsyncTTLinkSetup:
    """异步 TT 链路建链指示 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0051
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> AsyncTTLinkSetup:
        return cls(bytes(data))


# ===================================================================
# 0x0052 - 0x005C  超宽带脉冲测量/感知
# ===================================================================

@dataclass
class UWBMeasCapRequest(_ZeroPayload):
    """超宽带脉冲测量能力请求。"""
    DATA_TYPE_INDEX = 0x0052
    BYTE_LENGTH = 0


@dataclass
class UWBMeasCapResponse:
    """超宽带脉冲测量能力响应 (400 bits / 50 bytes)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0053
    BYTE_LENGTH = 50

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 50)[:50]

    @classmethod
    def unpack(cls, data: bytes) -> UWBMeasCapResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:50]))


@dataclass
class UWBMeasConfig:
    """超宽带脉冲测量配置 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0054
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> UWBMeasConfig:
        return cls(bytes(data))


@dataclass
class UWBMeasConfigFeedback:
    """超宽带脉冲测量配置反馈。"""
    config_index: int     # 8
    status: int           # 8

    DATA_TYPE_INDEX = 0x0055
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF, self.status & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> UWBMeasConfigFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1])


@dataclass
class UWBMeasReport:
    """超宽带脉冲测量信息上报 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0056
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> UWBMeasReport:
        return cls(bytes(data))


@dataclass
class UWBSensingCapRequest(_ZeroPayload):
    """超宽带脉冲感知能力请求。"""
    DATA_TYPE_INDEX = 0x0057
    BYTE_LENGTH = 0


@dataclass
class UWBSensingCapResponse:
    """超宽带脉冲感知能力响应 (408 bits / 51 bytes)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0058
    BYTE_LENGTH = 51

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 51)[:51]

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingCapResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:51]))


# ---------------------------------------------------------------------------
# 0x0059 超宽带脉冲感知配置 (可变长)
# ---------------------------------------------------------------------------

@dataclass
class UWBSensingConfig:
    """超宽带脉冲感知配置 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0059
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingConfig:
        return cls(bytes(data))


@dataclass
class UWBSensingConfigFeedback:
    """超宽带脉冲感知配置反馈。"""
    config_index: int
    status: int

    DATA_TYPE_INDEX = 0x005A
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF, self.status & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingConfigFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1])


@dataclass
class UWBSensingReport:
    """超宽带脉冲感知信息上报 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x005B
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingReport:
        return cls(bytes(data))


@dataclass
class UWBSensingAction:
    """超宽带脉冲感知行为指示。"""
    config_index: int
    start_slot: int          # 32
    action_config: int       # 8

    DATA_TYPE_INDEX = 0x005C
    BYTE_LENGTH = 6

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF]) + self.start_slot.to_bytes(4, "big") + bytes([self.action_config & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingAction:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], int.from_bytes(data[1:5], "big"), data[5])


# ===================================================================
# 0x005D - 0x005E  资源预留
# ===================================================================

@dataclass
class ResourceReservation:
    """资源预留指示。"""
    config_index: int             # 8
    effective_slot: int           # 32
    event_group_period: int       # 16
    event_period: int             # 16
    event_length: int             # 16
    event_count: int              # 8
    scheduling_slot: int          # 3

    DATA_TYPE_INDEX = 0x005D
    BYTE_LENGTH = 13

    def pack(self) -> bytes:
        buf = bytes([self.config_index & 0xFF])
        buf += self.effective_slot.to_bytes(4, "big")
        buf += struct.pack(">HHHB", self.event_group_period,
                           self.event_period, self.event_length,
                           self.event_count)
        buf += bytes([(self.scheduling_slot & 0x07) << 5])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> ResourceReservation:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        ci = data[0]
        es = int.from_bytes(data[1:5], "big")
        egp, ep, el, ec = struct.unpack(">HHHB", data[5:12])
        ss = (data[12] >> 5) & 0x07
        return cls(ci, es, egp, ep, el, ec, ss)


@dataclass
class ResourceReservationTerminate:
    """资源预留终止。"""
    config_index: int   # 8
    reason: int         # 8

    DATA_TYPE_INDEX = 0x005E
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF, self.reason & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> ResourceReservationTerminate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1])


# ===================================================================
# 0x005F - 0x0069  窄带跳频感知
# ===================================================================

@dataclass
class NarrowbandSensingRequest:
    """窄带跳频感知流程请求。"""
    payload: bytes  # 16 bytes

    DATA_TYPE_INDEX = 0x005F
    BYTE_LENGTH = 16

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 16)[:16]

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:16]))


@dataclass
class NarrowbandSensingFeedback:
    """窄带跳频感知流程反馈。"""
    process_index: int   # 8
    status: int          # 8

    DATA_TYPE_INDEX = 0x0060
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.process_index & 0xFF, self.status & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1])


@dataclass
class NarrowbandProxySensingRequest:
    """窄带跳频代理感知请求。"""
    proxy_index: int            # 8
    sensing_index: int          # 8
    meas_quantity: int          # 32
    report_period: int          # 32
    bandwidth: int              # 8

    DATA_TYPE_INDEX = 0x0061
    BYTE_LENGTH = 11

    def pack(self) -> bytes:
        buf = bytes([self.proxy_index & 0xFF, self.sensing_index & 0xFF])
        buf += self.meas_quantity.to_bytes(4, "big")
        buf += self.report_period.to_bytes(4, "big")
        buf += bytes([self.bandwidth & 0xFF])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandProxySensingRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(
            data[0], data[1],
            int.from_bytes(data[2:6], "big"),
            int.from_bytes(data[6:10], "big"),
            data[10],
        )


@dataclass
class NarrowbandProxySensingFeedback:
    """窄带跳频代理感知反馈。"""
    proxy_index: int
    sensing_index: int          # 16
    status: int                 # 8
    meas_quantity1: int         # 32
    meas_quantity2: int         # 32
    bandwidth1: int             # 8
    bandwidth2: int             # 8

    DATA_TYPE_INDEX = 0x0062
    BYTE_LENGTH = 14

    def pack(self) -> bytes:
        buf = bytes([self.proxy_index & 0xFF])
        buf += struct.pack(">HB", self.sensing_index, self.status)
        buf += self.meas_quantity1.to_bytes(4, "big")
        buf += self.meas_quantity2.to_bytes(4, "big")
        buf += bytes([self.bandwidth1 & 0xFF, self.bandwidth2 & 0xFF])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandProxySensingFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        pi = data[0]
        si, st = struct.unpack(">HB", data[1:4])
        mq1 = int.from_bytes(data[4:8], "big")
        mq2 = int.from_bytes(data[8:12], "big")
        return cls(pi, si, st, mq1, mq2, data[12], data[13])


@dataclass
class NarrowbandSensingCapRequest(_ZeroPayload):
    """窄带跳频感知能力请求。"""
    DATA_TYPE_INDEX = 0x0063
    BYTE_LENGTH = 0


@dataclass
class NarrowbandSensingCapResponse:
    """窄带跳频感知能力响应 (50 bytes)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0064
    BYTE_LENGTH = 50

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 50)[:50]

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingCapResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:50]))


@dataclass
class NarrowbandSensingConfig:
    """窄带跳频感知配置 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0065
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingConfig:
        return cls(bytes(data))


@dataclass
class NarrowbandSensingConfigFeedback:
    """窄带跳频感知配置反馈。"""
    config_index: int
    status: int

    DATA_TYPE_INDEX = 0x0066
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF, self.status & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingConfigFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1])


@dataclass
class SensingDeviceStatusReport:
    """感知设备状态上报。"""
    config_index: int     # 8
    stability: int        # 1 bit (存储在字节高位)

    DATA_TYPE_INDEX = 0x0067
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF, (self.stability & 1) << 7])

    @classmethod
    def unpack(cls, data: bytes) -> SensingDeviceStatusReport:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], (data[1] >> 7) & 1)


@dataclass
class NarrowbandSensingReport:
    """窄带跳频感知信息上报 (可变长)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x0068
    BYTE_LENGTH = 0

    def pack(self) -> bytes:
        return self.payload

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingReport:
        return cls(bytes(data))


@dataclass
class NarrowbandSensingAction:
    """窄带跳频感知行为指示。"""
    config_index: int
    start_slot: int
    action_config: int

    DATA_TYPE_INDEX = 0x0069
    BYTE_LENGTH = 6

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF]) + self.start_slot.to_bytes(4, "big") + bytes([self.action_config & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandSensingAction:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], int.from_bytes(data[1:5], "big"), data[5])


# ===================================================================
# 0x006A - 0x0070  配置更新 / UWB 感知
# ===================================================================

@dataclass
class NarrowbandMeasConfigUpdateRequest:
    """窄带跳频测量信号配置更新请求 (32 bytes)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x006A
    BYTE_LENGTH = 32

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 32)[:32]

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasConfigUpdateRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:32]))


@dataclass
class NarrowbandMeasConfigUpdateIndication:
    """窄带跳频测量信号配置更新指示 (与请求结构相同, 32 bytes)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x006B
    BYTE_LENGTH = 32

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 32)[:32]

    @classmethod
    def unpack(cls, data: bytes) -> NarrowbandMeasConfigUpdateIndication:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:32]))


@dataclass
class UWBSensingProcessRequest:
    """超宽带脉冲感知流程请求 (16 bytes)。"""
    payload: bytes

    DATA_TYPE_INDEX = 0x006C
    BYTE_LENGTH = 16

    def pack(self) -> bytes:
        return (self.payload + b"\x00" * 16)[:16]

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingProcessRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(bytes(data[:16]))


@dataclass
class UWBSensingProcessFeedback:
    """超宽带脉冲感知流程反馈。"""
    process_index: int
    status: int

    DATA_TYPE_INDEX = 0x006D
    BYTE_LENGTH = 2

    def pack(self) -> bytes:
        return bytes([self.process_index & 0xFF, self.status & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> UWBSensingProcessFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], data[1])


@dataclass
class UWBProxySensingRequest:
    """超宽带脉冲代理感知请求 (11 bytes)。"""
    proxy_index: int
    sensing_index: int
    meas_quantity: int
    report_period: int
    bandwidth: int

    DATA_TYPE_INDEX = 0x006E
    BYTE_LENGTH = 11

    def pack(self) -> bytes:
        buf = bytes([self.proxy_index & 0xFF, self.sensing_index & 0xFF])
        buf += self.meas_quantity.to_bytes(4, "big")
        buf += self.report_period.to_bytes(4, "big")
        buf += bytes([self.bandwidth & 0xFF])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> UWBProxySensingRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(
            data[0], data[1],
            int.from_bytes(data[2:6], "big"),
            int.from_bytes(data[6:10], "big"),
            data[10],
        )


@dataclass
class UWBProxySensingFeedback:
    """超宽带脉冲代理感知反馈 (14 bytes)。"""
    proxy_index: int
    sensing_index: int       # 16
    status: int
    meas_quantity1: int      # 32
    meas_quantity2: int      # 32
    bandwidth1: int
    bandwidth2: int

    DATA_TYPE_INDEX = 0x006F
    BYTE_LENGTH = 14

    def pack(self) -> bytes:
        buf = bytes([self.proxy_index & 0xFF])
        buf += struct.pack(">HB", self.sensing_index, self.status)
        buf += self.meas_quantity1.to_bytes(4, "big")
        buf += self.meas_quantity2.to_bytes(4, "big")
        buf += bytes([self.bandwidth1 & 0xFF, self.bandwidth2 & 0xFF])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> UWBProxySensingFeedback:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        pi = data[0]
        si, st = struct.unpack(">HB", data[1:4])
        mq1 = int.from_bytes(data[4:8], "big")
        mq2 = int.from_bytes(data[8:12], "big")
        return cls(pi, si, st, mq1, mq2, data[12], data[13])


@dataclass
class UWBMeasAction:
    """超宽带脉冲测量行为指示。"""
    config_index: int
    start_slot: int
    action_config: int

    DATA_TYPE_INDEX = 0x0070
    BYTE_LENGTH = 6

    def pack(self) -> bytes:
        return bytes([self.config_index & 0xFF]) + self.start_slot.to_bytes(4, "big") + bytes([self.action_config & 0xFF])

    @classmethod
    def unpack(cls, data: bytes) -> UWBMeasAction:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[0], int.from_bytes(data[1:5], "big"), data[5])
