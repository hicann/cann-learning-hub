"""系统管理帧编解码 -- TXS-10002-2025 标准 6.6

实现系统管理帧 (System Management Frame) 的完整编解码:
- SMFHeader: 系统管理帧数据包头 (分段指示 + 信令编号)
- ScheduleSignaling: 系统管理帧调度信令 (6.6.2.1)
- LinkSignaling: 链路信令 (6.6.2.2)
- OffsetSignaling: 偏移信令 (6.6.2.3)
- SystemManagementFrame: 完整系统管理帧组装与解析
"""

from __future__ import annotations

__all__ = [
    "FrameTypeConfig",
    "LinkSignaling",
    "OffsetSignaling",
    "OffsetUnit",
    "SMFHeader",
    "SMFSignalingTLV",
    "SMFSignalingType",
    "ScheduleSignaling",
    "ScheduleSlotLength",
    "SegmentIndication",
    "SystemManagementFrame",
    "TimeResourceEntry",
    "reassemble_smf",
]


import struct
from dataclasses import dataclass, field
from enum import IntEnum

# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------


class SegmentIndication(IntEnum):
    """分段指示 (6.6.2)"""
    COMPLETE = 0       # 不分段的完整数据包
    FIRST = 1          # 分段的首个系统管理包
    MIDDLE = 2         # 分段的中间数据包
    LAST = 3           # 分段的最后一个数据包


class SMFSignalingType(IntEnum):
    """信令类型 (6.6.2)"""
    SCHEDULE = 0       # 系统管理帧调度信令
    LINK = 1           # 链路信令
    OFFSET = 2         # 偏移信令 (标准中称"扩展参数信令")


class FrameTypeConfig(IntEnum):
    """无线帧类型配置 (6.6.2.1, 表17)"""
    FT1 = 0
    FT2 = 1
    FT3_M0 = 2
    FT3_M1 = 3
    FT3_M2 = 4
    FT3_M3 = 5
    FT3_M4 = 6
    FT3_M5 = 7
    FT4_M0 = 8
    FT4_M1 = 9
    FT4_M2 = 10
    FT4_M3 = 11
    FT4_M4 = 12
    FT4_M5 = 13


class ScheduleSlotLength(IntEnum):
    """调度时隙长度 (6.6.2.2)"""
    US_25 = 0
    US_50 = 1
    US_75 = 2
    US_100 = 3
    US_125 = 4    # 默认值


class OffsetUnit(IntEnum):
    """偏移量单位 (6.6.2.3)"""
    US_25 = 0
    US_300 = 1


# ---------------------------------------------------------------------------
# 6.6.2 系统管理帧数据包头
# ---------------------------------------------------------------------------


@dataclass
class SMFHeader:
    """系统管理帧数据包头

    1 字节:
    - 分段指示: 2 bits (高位)
    - 信令编号: 6 bits (低位)
    """
    segment_indication: int = 0    # 0-3
    signaling_number: int = 0      # 0-63

    BYTE_LENGTH = 1

    def pack(self) -> bytes:
        b = ((self.segment_indication & 0x03) << 6) | (self.signaling_number & 0x3F)
        return bytes([b])

    @classmethod
    def unpack(cls, data: bytes) -> SMFHeader:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError("数据不足: 需要 1 字节")
        b = data[0]
        return cls(
            segment_indication=(b >> 6) & 0x03,
            signaling_number=b & 0x3F,
        )


# ---------------------------------------------------------------------------
# 6.6.2.1 系统管理帧调度信令
# ---------------------------------------------------------------------------


@dataclass
class ScheduleSignaling:
    """系统管理帧调度信令 (6.6.2.1, 表16)

    指示系统管理帧的间隔、带宽、频点表以及参数生效时刻。
    """
    effective_slot: int = 0       # 信令生效时隙 (4 字节, 基础时隙单位)
    interval: int = 0             # 系统管理帧间隔 (2 字节, 基础时隙单位)
    frame_type: int = 0           # 无线帧类型指示 (4 bits)
    bandwidth: int = 0            # 带宽指示 (2 bits, 0=1M, 1=2M, 2=4M)
    pilot_density: int = 0        # 导频密度指示 (2 bits)
    channel_count: int = 0        # 可用频点个数 (1 字节)
    channel_table: bytes = b""    # 频点表 (channel_count 字节)

    def pack(self) -> bytes:
        buf = struct.pack(">I", self.effective_slot & 0xFFFFFFFF)
        buf += struct.pack(">H", self.interval & 0xFFFF)
        config = (
            ((self.frame_type & 0x0F) << 4)
            | ((self.bandwidth & 0x03) << 2)
            | (self.pilot_density & 0x03)
        )
        buf += bytes([config])
        buf += bytes([self.channel_count & 0xFF])
        buf += bytes(self.channel_table[:self.channel_count])
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> ScheduleSignaling:
        if len(data) < 8:
            raise ValueError("调度信令数据不足: 至少需要 8 字节")
        eff = struct.unpack(">I", data[0:4])[0]
        interval = struct.unpack(">H", data[4:6])[0]
        config = data[6]
        ft = (config >> 4) & 0x0F
        bw = (config >> 2) & 0x03
        pd = config & 0x03
        ch_count = data[7]
        ch_table = bytes(data[8:8 + ch_count])
        return cls(eff, interval, ft, bw, pd, ch_count, ch_table)


# ---------------------------------------------------------------------------
# 6.6.2.2 链路信令 -- 时间资源条目
# ---------------------------------------------------------------------------


@dataclass
class TimeResourceEntry:
    """单个时间资源配置条目 (6.6.2.2)

    描述一个时间分片的资源分配:
    - 偏移量       (2 字节, 调度时隙单位)
    - 持续长度     (2 字节, 调度时隙单位)
    - 周期间隔     (2 字节, 调度时隙单位)
    - 重复次数     (1 字节)
    """
    offset: int = 0               # 时间片偏移量
    duration: int = 0             # 时间片持续长度
    period: int = 0               # 时间片周期间隔
    repeat_count: int = 0         # 重复次数

    BYTE_LENGTH = 7

    def pack(self) -> bytes:
        return struct.pack(">HHHB",
                           self.offset & 0xFFFF,
                           self.duration & 0xFFFF,
                           self.period & 0xFFFF,
                           self.repeat_count & 0xFF)

    @classmethod
    def unpack(cls, data: bytes) -> TimeResourceEntry:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError("时间资源条目数据不足: 需要 7 字节")
        off, dur, per, rep = struct.unpack(">HHHB", data[:7])
        return cls(off, dur, per, rep)


# ---------------------------------------------------------------------------
# 6.6.2.2 链路信令
# ---------------------------------------------------------------------------


@dataclass
class LinkSignaling:
    """链路信令 (6.6.2.2, 表18)

    描述链路的时间资源分布情况。
    """
    llid: int = 0                              # 逻辑链路标识 (3 字节)
    effective_slot: int = 0                    # 信令生效时隙 (4 字节)
    link_period_factor: int = 0                # 链路周期因子 (1 字节)
    schedule_slot_length: int = 4              # 调度时隙长度 (3 bits, 默认 125us)
    resources: list[TimeResourceEntry] = field(default_factory=list)

    def pack(self) -> bytes:
        res_count = len(self.resources)
        buf = self.llid.to_bytes(3, "big")
        buf += struct.pack(">I", self.effective_slot & 0xFFFFFFFF)
        buf += bytes([self.link_period_factor & 0xFF])
        # 时间资源配置个数 (5 bits) + 调度时隙长度 (3 bits) 合为 1 字节
        config = ((res_count & 0x1F) << 3) | (self.schedule_slot_length & 0x07)
        buf += bytes([config])
        for entry in self.resources:
            buf += entry.pack()
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> LinkSignaling:
        if len(data) < 9:
            raise ValueError("链路信令数据不足: 至少需要 9 字节")
        llid = int.from_bytes(data[0:3], "big")
        eff = struct.unpack(">I", data[3:7])[0]
        lpf = data[7]
        config = data[8]
        res_count = (config >> 3) & 0x1F
        slot_len = config & 0x07
        entries: list[TimeResourceEntry] = []
        pos = 9
        for _ in range(res_count):
            if pos + TimeResourceEntry.BYTE_LENGTH > len(data):
                raise ValueError("时间资源条目数据截断")
            entries.append(TimeResourceEntry.unpack(data[pos:]))
            pos += TimeResourceEntry.BYTE_LENGTH
        return cls(llid, eff, lpf, slot_len, entries)


# ---------------------------------------------------------------------------
# 6.6.2.3 偏移信令
# ---------------------------------------------------------------------------


@dataclass
class OffsetSignaling:
    """偏移信令 (6.6.2.3, 表19)

    标识链路建立和同步的偏移量信息。
    """
    llid: int = 0                  # 逻辑链路标识 (3 字节)
    offset: int = 0                # 偏移量 (15 bits)
    offset_unit: int = 0           # 偏移量单位 (1 bit, 0=25us, 1=300us)

    BYTE_LENGTH = 5

    def pack(self) -> bytes:
        buf = self.llid.to_bytes(3, "big")
        # 偏移量 15 bits + 偏移量单位 1 bit => 2 字节
        val = ((self.offset & 0x7FFF) << 1) | (self.offset_unit & 0x01)
        buf += struct.pack(">H", val)
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> OffsetSignaling:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError("偏移信令数据不足: 需要 5 字节")
        llid = int.from_bytes(data[0:3], "big")
        val = struct.unpack(">H", data[3:5])[0]
        offset = (val >> 1) & 0x7FFF
        offset_unit = val & 0x01
        return cls(llid, offset, offset_unit)


# ---------------------------------------------------------------------------
# 信令 TLV 封装
# ---------------------------------------------------------------------------


@dataclass
class SMFSignalingTLV:
    """系统管理帧的信令 TLV 结构 (类型 + 长度 + 内容)"""
    sig_type: int = 0              # 信令类型 (1 字节)
    content: bytes = b""           # 信令内容 (0~255 字节)

    def pack(self) -> bytes:
        length = len(self.content)
        if length > 255:
            raise ValueError(f"信令内容超出最大长度: {length} > 255")
        return bytes([self.sig_type & 0xFF, length & 0xFF]) + self.content

    @classmethod
    def unpack(cls, data: bytes) -> tuple[SMFSignalingTLV, int]:
        """解析一个 TLV, 返回 (对象, 消耗字节数)"""
        if len(data) < 2:
            raise ValueError("TLV 数据不足")
        sig_type = data[0]
        length = data[1]
        if len(data) < 2 + length:
            raise ValueError(f"TLV 内容不足: 需要 {length} 字节")
        content = bytes(data[2:2 + length])
        return cls(sig_type, content), 2 + length


# ---------------------------------------------------------------------------
# 系统管理帧完整结构
# ---------------------------------------------------------------------------


@dataclass
class SystemManagementFrame:
    """系统管理帧完整结构 (6.6.2)

    组合 SMFHeader + 多个 SMFSignalingTLV 构成完整系统管理帧载荷。
    提供高层便捷方法, 直接操作 ScheduleSignaling / LinkSignaling / OffsetSignaling。
    """
    header: SMFHeader = field(default_factory=SMFHeader)
    signalings: list[SMFSignalingTLV] = field(default_factory=list)

    def pack(self) -> bytes:
        buf = self.header.pack()
        for sig in self.signalings:
            buf += sig.pack()
        return buf

    @classmethod
    def unpack(cls, data: bytes) -> SystemManagementFrame:
        if len(data) < 1:
            raise ValueError("系统管理帧数据为空")
        header = SMFHeader.unpack(data)
        signalings: list[SMFSignalingTLV] = []
        pos = SMFHeader.BYTE_LENGTH
        while pos < len(data):
            if pos + 2 > len(data):
                break
            tlv, consumed = SMFSignalingTLV.unpack(data[pos:])
            signalings.append(tlv)
            pos += consumed
        return cls(header, signalings)

    # ---- 便捷构建方法 ----

    def add_schedule(self, sig: ScheduleSignaling) -> None:
        """添加调度信令"""
        self.signalings.append(SMFSignalingTLV(
            sig_type=SMFSignalingType.SCHEDULE,
            content=sig.pack(),
        ))

    def add_link(self, sig: LinkSignaling) -> None:
        """添加链路信令"""
        self.signalings.append(SMFSignalingTLV(
            sig_type=SMFSignalingType.LINK,
            content=sig.pack(),
        ))

    def add_offset(self, sig: OffsetSignaling) -> None:
        """添加偏移信令"""
        self.signalings.append(SMFSignalingTLV(
            sig_type=SMFSignalingType.OFFSET,
            content=sig.pack(),
        ))

    def get_schedules(self) -> list[ScheduleSignaling]:
        """提取所有调度信令"""
        return [
            ScheduleSignaling.unpack(tlv.content)
            for tlv in self.signalings
            if tlv.sig_type == SMFSignalingType.SCHEDULE
        ]

    def get_links(self) -> list[LinkSignaling]:
        """提取所有链路信令"""
        return [
            LinkSignaling.unpack(tlv.content)
            for tlv in self.signalings
            if tlv.sig_type == SMFSignalingType.LINK
        ]

    def get_offsets(self) -> list[OffsetSignaling]:
        """提取所有偏移信令"""
        return [
            OffsetSignaling.unpack(tlv.content)
            for tlv in self.signalings
            if tlv.sig_type == SMFSignalingType.OFFSET
        ]


# ---------------------------------------------------------------------------
# 系统管理帧分段重组
# ---------------------------------------------------------------------------


def reassemble_smf(fragments: list[bytes]) -> SystemManagementFrame:
    """将分段的系统管理帧数据重组为完整帧

    每个 fragment 是一个完整的系统管理帧 payload (含 SMFHeader)。
    按照分段指示规则:
    - FIRST(1): 首段, 取其 header 的信令编号
    - MIDDLE(2): 中间段
    - LAST(3): 尾段
    - COMPLETE(0): 不分段, 直接解析
    """
    if not fragments:
        raise ValueError("没有可重组的分段数据")

    if len(fragments) == 1:
        return SystemManagementFrame.unpack(fragments[0])

    # 多段重组: 提取首段 header, 将所有段的 TLV 数据级联
    first_header = SMFHeader.unpack(fragments[0])
    combined_payload = b""
    for frag in fragments:
        # 跳过每段的 1 字节 header, 只取 TLV 部分
        combined_payload += frag[SMFHeader.BYTE_LENGTH:]

    # 用首段 header + 合并后的 TLV 数据重建
    full_data = first_header.pack() + combined_payload
    return SystemManagementFrame.unpack(full_data)
