"""功率控制信令 -- TXS-10002-2025 标准 7.2.13 / 7.3.2.27-29

提供三类功率控制信令的编解码以及功率控制状态管理:
- PowerControlRequest  (0x0019, 24 bits)
- PowerControlResponse (0x001A, 32 bits)
- PowerChangeIndication(0x001B, 32 bits)
"""

from __future__ import annotations

__all__ = [
    "TX_POWER_STOP_MANAGEMENT",
    "TX_POWER_UNAVAILABLE",
    "Bandwidth",
    "FreqDensity",
    "PowerChangeIndication",
    "PowerControlRequest",
    "PowerControlResponse",
    "PowerController",
]


import struct
from dataclasses import dataclass
from enum import IntEnum

# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class Bandwidth(IntEnum):
    BW_1MHZ = 0
    BW_2MHZ = 1
    BW_4MHZ = 2
    RESERVED = 3


class FreqDensity(IntEnum):
    RATIO_4_1 = 0
    RATIO_8_1 = 1
    RATIO_16_1 = 2
    NO_PILOT = 3


# 特殊发射功率取值
TX_POWER_UNAVAILABLE = 127
TX_POWER_STOP_MANAGEMENT = 126


# ---------------------------------------------------------------------------
# 7.3.2.27 功率控制请求 (data_type_index = 0x0019, 24 bits / 3 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PowerControlRequest:
    """功率控制请求信令。

    字段顺序 (MSB → LSB 按标准比特排列):
    - frame_type:       4 bits, 无线帧类型指示 (0-15)
    - bandwidth:        2 bits, 带宽指示 (Bandwidth 枚举)
    - freq_density:     2 bits, 频密度指示 (FreqDensity 枚举)
    - tx_power_change:  8 bits, 有符号, 要求接收方发射功率变化值 (dB)
    - sender_tx_power:  8 bits, 有符号, 发送方发射功率 (dBm)
    """
    frame_type: int
    bandwidth: int
    freq_density: int
    tx_power_change: int   # -128 ~ 127, 单位 dB
    sender_tx_power: int   # -127 ~ 20, 127 表示不可用

    DATA_TYPE_INDEX = 0x0019
    BYTE_LENGTH = 3

    def pack(self) -> bytes:
        """编码为 3 字节。"""
        b0 = ((self.frame_type & 0x0F) << 4) | ((self.bandwidth & 0x03) << 2) | (self.freq_density & 0x03)
        b1 = struct.pack("b", self.tx_power_change)[0]
        b2 = struct.pack("b", self.sender_tx_power)[0]
        return bytes([b0, b1, b2])

    @classmethod
    def unpack(cls, data: bytes) -> PowerControlRequest:
        """从 3 字节解码。"""
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据长度不足: 需要 {cls.BYTE_LENGTH} 字节, 实际 {len(data)}")
        frame_type = (data[0] >> 4) & 0x0F
        bandwidth = (data[0] >> 2) & 0x03
        freq_density = data[0] & 0x03
        tx_power_change = struct.unpack("b", bytes([data[1]]))[0]
        sender_tx_power = struct.unpack("b", bytes([data[2]]))[0]
        return cls(frame_type, bandwidth, freq_density, tx_power_change, sender_tx_power)


# ---------------------------------------------------------------------------
# 7.3.2.28 功率控制响应 (data_type_index = 0x001A, 32 bits / 4 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PowerControlResponse:
    """功率控制响应信令。

    字段顺序:
    - sender_min_power:          1 bit, 置 1 表示发送端以最小功率发送
    - sender_max_power:          1 bit, 置 1 表示发送端以最大功率发送
    - reserved:                  6 bits
    - tx_power_change:           8 bits, 有符号, 本端实际功率变化值 (dB)
    - sender_tx_power:           8 bits, 有符号, 发送端发射功率 (dBm)
    - acceptable_power_reduction: 8 bits, 无符号, 可接受的功率减少量 (dB)
    """
    sender_min_power: bool
    sender_max_power: bool
    tx_power_change: int    # -128 ~ 127, 单位 dB
    sender_tx_power: int    # -127 ~ 20, 127=不可用, 126=停止管理
    acceptable_power_reduction: int  # 0 ~ 255, 单位 dB

    DATA_TYPE_INDEX = 0x001A
    BYTE_LENGTH = 4

    def pack(self) -> bytes:
        """编码为 4 字节。"""
        b0 = ((int(self.sender_min_power) & 1) << 7) | ((int(self.sender_max_power) & 1) << 6)
        b1 = struct.pack("b", self.tx_power_change)[0]
        b2 = struct.pack("b", self.sender_tx_power)[0]
        b3 = self.acceptable_power_reduction & 0xFF
        return bytes([b0, b1, b2, b3])

    @classmethod
    def unpack(cls, data: bytes) -> PowerControlResponse:
        """从 4 字节解码。"""
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据长度不足: 需要 {cls.BYTE_LENGTH} 字节, 实际 {len(data)}")
        sender_min_power = bool((data[0] >> 7) & 1)
        sender_max_power = bool((data[0] >> 6) & 1)
        tx_power_change = struct.unpack("b", bytes([data[1]]))[0]
        sender_tx_power = struct.unpack("b", bytes([data[2]]))[0]
        acceptable_power_reduction = data[3]
        return cls(sender_min_power, sender_max_power, tx_power_change,
                   sender_tx_power, acceptable_power_reduction)


# ---------------------------------------------------------------------------
# 7.3.2.29 功率变化指示 (data_type_index = 0x001B, 32 bits / 4 bytes)
# ---------------------------------------------------------------------------

@dataclass
class PowerChangeIndication:
    """功率变化指示信令。

    字段顺序:
    - frame_type:        4 bits, 无线帧类型指示
    - bandwidth:         2 bits, 带宽指示
    - freq_density:      2 bits, 导频密度指示
    - sender_min_power:  1 bit
    - sender_max_power:  1 bit
    - reserved:          6 bits
    - tx_power_change:   8 bits, 有符号, 本次功率变化值 (dB)
    - sender_tx_power:   8 bits, 有符号, 变更后发射功率 (dBm)
    """
    frame_type: int
    bandwidth: int
    freq_density: int
    sender_min_power: bool
    sender_max_power: bool
    tx_power_change: int   # -128 ~ 127
    sender_tx_power: int   # -127 ~ 20, 127=不可用, 126=停止管理

    DATA_TYPE_INDEX = 0x001B
    BYTE_LENGTH = 4

    def pack(self) -> bytes:
        """编码为 4 字节。"""
        b0 = ((self.frame_type & 0x0F) << 4) | ((self.bandwidth & 0x03) << 2) | (self.freq_density & 0x03)
        b1 = ((int(self.sender_min_power) & 1) << 7) | ((int(self.sender_max_power) & 1) << 6)
        b2 = struct.pack("b", self.tx_power_change)[0]
        b3 = struct.pack("b", self.sender_tx_power)[0]
        return bytes([b0, b1, b2, b3])

    @classmethod
    def unpack(cls, data: bytes) -> PowerChangeIndication:
        """从 4 字节解码。"""
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据长度不足: 需要 {cls.BYTE_LENGTH} 字节, 实际 {len(data)}")
        frame_type = (data[0] >> 4) & 0x0F
        bandwidth = (data[0] >> 2) & 0x03
        freq_density = data[0] & 0x03
        sender_min_power = bool((data[1] >> 7) & 1)
        sender_max_power = bool((data[1] >> 6) & 1)
        tx_power_change = struct.unpack("b", bytes([data[2]]))[0]
        sender_tx_power = struct.unpack("b", bytes([data[3]]))[0]
        return cls(frame_type, bandwidth, freq_density,
                   sender_min_power, sender_max_power,
                   tx_power_change, sender_tx_power)


# ---------------------------------------------------------------------------
# 功率控制状态管理器 -- 7.2.13 四种场景
# ---------------------------------------------------------------------------

class PowerController:
    """单链路功率控制状态管理。

    管理本端发射功率, 处理对端发来的功率控制请求/响应/变化指示,
    并生成需要发送给对端的信令。

    参数
    ----
    tx_power_dbm : 当前发射功率 (dBm)
    min_power_dbm : 最小支持发射功率 (dBm)
    max_power_dbm : 最大支持发射功率 (dBm)
    """

    def __init__(
        self,
        tx_power_dbm: float = 0.0,
        min_power_dbm: float = -127.0,
        max_power_dbm: float = 20.0,
    ):
        self.tx_power_dbm = tx_power_dbm
        self.min_power_dbm = min_power_dbm
        self.max_power_dbm = max_power_dbm
        self._power_management_active = False

    @property
    def is_at_min_power(self) -> bool:
        return self.tx_power_dbm <= self.min_power_dbm

    @property
    def is_at_max_power(self) -> bool:
        return self.tx_power_dbm >= self.max_power_dbm

    @property
    def acceptable_power_reduction(self) -> int:
        """可接受的功率减少量 (dB), 0-255 范围。"""
        reduction = self.tx_power_dbm - self.min_power_dbm
        return max(0, min(255, int(reduction)))

    def _clamp_power(self, power_dbm: float) -> float:
        return max(self.min_power_dbm, min(self.max_power_dbm, power_dbm))

    # -- 场景 a: 处理对端请求调整本端发射功率 --

    def handle_request(self, req: PowerControlRequest) -> PowerControlResponse:
        """处理对端的功率控制请求, 调整本端功率并返回响应。"""
        old_power = self.tx_power_dbm
        self.tx_power_dbm = self._clamp_power(old_power + req.tx_power_change)
        actual_change = int(self.tx_power_dbm - old_power)
        return PowerControlResponse(
            sender_min_power=self.is_at_min_power,
            sender_max_power=self.is_at_max_power,
            tx_power_change=actual_change,
            sender_tx_power=int(self.tx_power_dbm),
            acceptable_power_reduction=self.acceptable_power_reduction,
        )

    # -- 场景 b: 生成请求, 用于查询对端可减少量 --

    def create_request(
        self,
        requested_change_db: int,
        frame_type: int = 0,
        bandwidth: int = Bandwidth.BW_1MHZ,
        freq_density: int = FreqDensity.RATIO_4_1,
    ) -> PowerControlRequest:
        """生成功率控制请求信令。"""
        return PowerControlRequest(
            frame_type=frame_type,
            bandwidth=bandwidth,
            freq_density=freq_density,
            tx_power_change=requested_change_db,
            sender_tx_power=int(self.tx_power_dbm),
        )

    def apply_response(self, resp: PowerControlResponse) -> None:
        """处理对端的功率控制响应, 可据此调整本端策略。

        场景 b: 根据对端反馈的 acceptable_power_reduction 调整本端功率。
        """
        pass

    # -- 场景 c: 功率等级管理 --

    def start_power_management(self) -> None:
        """启动功率等级管理。"""
        self._power_management_active = True

    def stop_power_management(self) -> None:
        """停止功率等级管理。"""
        self._power_management_active = False

    @property
    def power_management_active(self) -> bool:
        return self._power_management_active

    def create_change_indication(
        self,
        actual_change_db: int,
        frame_type: int = 0,
        bandwidth: int = Bandwidth.BW_1MHZ,
        freq_density: int = FreqDensity.RATIO_4_1,
    ) -> PowerChangeIndication:
        """生成功率变化指示信令 (场景 c)。"""
        return PowerChangeIndication(
            frame_type=frame_type,
            bandwidth=bandwidth,
            freq_density=freq_density,
            sender_min_power=self.is_at_min_power,
            sender_max_power=self.is_at_max_power,
            tx_power_change=actual_change_db,
            sender_tx_power=int(self.tx_power_dbm),
        )

    # -- 场景 d: 查询功率等级 --

    def create_query_request(
        self,
        frame_type: int = 0,
        bandwidth: int = Bandwidth.BW_1MHZ,
        freq_density: int = FreqDensity.RATIO_4_1,
    ) -> PowerControlRequest:
        """生成功率查询请求 (tx_power_change=0)。"""
        return PowerControlRequest(
            frame_type=frame_type,
            bandwidth=bandwidth,
            freq_density=freq_density,
            tx_power_change=0,
            sender_tx_power=int(self.tx_power_dbm),
        )
