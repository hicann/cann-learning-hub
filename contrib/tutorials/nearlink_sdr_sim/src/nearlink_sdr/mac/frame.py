"""数据链路层控制面帧结构 -- TXS-10002-2025 标准 7.3.2

控制面帧由数据类型索引 + 数据长度 + 控制面数据组成。
支持信令消息的注册、编码和解码。
"""

from __future__ import annotations

__all__ = [
    "AsyncDataFrame",
    "ControlFrame",
    "MuxFrame",
    "SegmentType",
    "SyncDataFrame",
]


from dataclasses import dataclass
from enum import IntEnum


class SegmentType(IntEnum):
    """分段类型指示 (7.3.3.2)。"""
    COMPLETE = 0      # 完整单一分段
    FIRST = 1         # SDU 第一个分段
    MIDDLE = 2        # SDU 中间分段
    LAST = 3          # SDU 最后分段


# ---------------------------------------------------------------------------
# 7.3.2.1 控制面帧通用结构
# ---------------------------------------------------------------------------

@dataclass
class ControlFrame:
    """控制面帧: [data_type_index: 2B][data_length: 1B][payload: NB]"""
    data_type_index: int   # 16 bits (0x0000 - 0xFFFF)
    payload: bytes

    def pack(self) -> bytes:
        """编码为字节流。"""
        header = self.data_type_index.to_bytes(2, "big")
        length = len(self.payload).to_bytes(1, "big")
        return header + length + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> tuple[ControlFrame, int]:
        """从字节流解码, 返回 (帧, 已消费字节数)。"""
        if len(data) < 3:
            raise ValueError(f"控制面帧头部不足: 需要 3 字节, 实际 {len(data)}")
        data_type_index = int.from_bytes(data[0:2], "big")
        payload_len = data[2]
        total = 3 + payload_len
        if len(data) < total:
            raise ValueError(f"控制面帧数据不足: 需要 {total} 字节, 实际 {len(data)}")
        payload = data[3:total]
        return cls(data_type_index, payload), total


# ---------------------------------------------------------------------------
# 7.3.3.2 异步数据链路帧
# ---------------------------------------------------------------------------

@dataclass
class AsyncDataFrame:
    """异步数据帧: [分段类型: 2b][数据长度: 11b][预留: 3b][数据: NB]"""
    segment_type: int   # 2 bits
    data: bytes

    def pack(self) -> bytes:
        """编码为字节流 (不含数据包类型索引前缀)。"""
        length = len(self.data)
        if length > 2047:
            raise ValueError(f"数据长度超过 2047 字节: {length}")
        # byte0: reserved(2b) + segment_type(2b) + data_length_high(4b)
        # byte1: data_length_low(7b) + reserved(1b)
        b0 = ((self.segment_type & 0x03) << 4) | ((length >> 7) & 0x0F)
        b1 = ((length & 0x7F) << 1)
        return bytes([b0, b1]) + self.data

    @classmethod
    def unpack(cls, data: bytes) -> AsyncDataFrame:
        """从字节流解码 (不含数据包类型索引)。"""
        if len(data) < 2:
            raise ValueError(f"异步数据帧头部不足: 需要 2 字节, 实际 {len(data)}")
        segment_type = (data[0] >> 4) & 0x03
        length = ((data[0] & 0x0F) << 7) | ((data[1] >> 1) & 0x7F)
        payload = data[2:2 + length]
        if len(payload) < length:
            raise ValueError(f"异步数据帧数据不足: 需要 {length} 字节, 实际 {len(payload)}")
        return cls(segment_type, payload)


# ---------------------------------------------------------------------------
# 7.3.3.3 链接态同步数据链路帧
# ---------------------------------------------------------------------------

@dataclass
class SyncDataFrame:
    """链接态同步数据帧 (非周期适配)。"""
    pdu_seq: int          # 5 bits, PDU 顺序号
    event_group: int      # 8 bits, 事件组编号
    frame_format: int     # 1 bit, 0=周期适配, 1=非周期适配
    segment_type: int     # 2 bits
    data: bytes
    time_offset: int | None = None    # 24 bits, 微秒, 首段/单一分段
    sdu_seq: int | None = None        # 5 bits, 仅周期适配

    def pack(self) -> bytes:
        """编码为字节流 (不含数据包类型索引前缀)。"""
        length = len(self.data)
        if length > 2047:
            raise ValueError(f"数据长度超过 2047 字节: {length}")
        # byte0: pdu_seq(5b) + reserved(3b)
        b0 = (self.pdu_seq & 0x1F) << 3
        # byte1: event_group(8b)
        b1 = self.event_group & 0xFF
        # byte2: frame_format(1b) + segment_type(2b) + data_length_high(5b)
        b2 = ((self.frame_format & 1) << 7) | ((self.segment_type & 0x03) << 5) | ((length >> 6) & 0x1F)
        # byte3: data_length_low(6b) + reserved(2b)
        b3 = ((length & 0x3F) << 2)
        result = bytes([b0, b1, b2, b3])

        # 时间偏移量 (首段和单一分段, 非周期适配)
        if self.frame_format == 1 and self.segment_type in (SegmentType.COMPLETE, SegmentType.FIRST):
            offset = self.time_offset if self.time_offset is not None else 0
            result += offset.to_bytes(3, "big")

        # SDU 顺序号 (周期适配)
        if self.frame_format == 0:
            seq = self.sdu_seq if self.sdu_seq is not None else 0
            result += bytes([(seq & 0x1F) << 3])

        result += self.data
        return result

    @classmethod
    def unpack(cls, data: bytes) -> SyncDataFrame:
        """从字节流解码 (不含数据包类型索引)。"""
        if len(data) < 4:
            raise ValueError(f"同步数据帧头部不足: 需要 4 字节, 实际 {len(data)}")
        pdu_seq = (data[0] >> 3) & 0x1F
        event_group = data[1]
        frame_format = (data[2] >> 7) & 1
        segment_type = (data[2] >> 5) & 0x03
        length = ((data[2] & 0x1F) << 6) | ((data[3] >> 2) & 0x3F)

        offset = 4
        time_offset = None
        sdu_seq = None

        if frame_format == 1 and segment_type in (SegmentType.COMPLETE, SegmentType.FIRST):
            if len(data) < offset + 3:
                raise ValueError("时间偏移量数据不足")
            time_offset = int.from_bytes(data[offset:offset + 3], "big")
            offset += 3

        if frame_format == 0:
            if len(data) < offset + 1:
                raise ValueError("SDU 顺序号数据不足")
            sdu_seq = (data[offset] >> 3) & 0x1F
            offset += 1

        payload = data[offset:offset + length]
        if len(payload) < length:
            raise ValueError(f"同步数据帧数据不足: 需要 {length} 字节, 实际 {len(payload)}")
        return cls(pdu_seq, event_group, frame_format, segment_type,
                   payload, time_offset, sdu_seq)


# ---------------------------------------------------------------------------
# 7.3.4 数据面与控制面复用帧
# ---------------------------------------------------------------------------

@dataclass
class MuxFrame:
    """数据面与控制面复用帧。

    存储一组控制面帧和一个可选数据帧 (异步或同步)。
    """
    control_frames: list[ControlFrame]
    data_frame: AsyncDataFrame | SyncDataFrame | None = None

    def pack(self) -> bytes:
        """编码全部帧为字节流。"""
        result = bytearray()
        for cf in self.control_frames:
            result.extend(cf.pack())
        if self.data_frame is not None:
            result.extend(self.data_frame.pack())
        return bytes(result)
