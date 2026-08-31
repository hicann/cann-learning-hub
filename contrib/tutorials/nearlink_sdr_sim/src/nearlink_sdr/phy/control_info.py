"""物理层控制信息 -- TXS-10002-2025 标准 6.4

A 组: 无线帧类型 1 和 2, CRC12, 种子为同步序列低 12 位。
B 组: 无线帧类型 3 和 4, CRC24B, 种子 0x555555, CRC 与 LLID 异或。
"""

from __future__ import annotations

__all__ = [
    "ControlInfoA1",
    "ControlInfoA2",
    "ControlInfoA3",
    "ControlInfoA4",
    "ControlInfoA5",
    "ControlInfoA6",
    "ControlInfoA7",
    "ControlInfoB1",
    "ControlInfoB2",
    "ControlInfoB3",
    "ControlInfoB4",
    "ControlInfoB5",
    "ControlInfoType",
    "polar_decode_control",
    "polar_encode_control",
]


from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from nearlink_sdr.common.crc import (
    CRC12_POLY,
    CRC24B_POLY,
    crc_attach,
    crc_check,
)

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def _int_to_bits(value: int, width: int) -> NDArray[np.int_]:
    """将无符号整数转为低位在前 (LSB-first) 的比特数组。"""
    bits = np.zeros(width, dtype=int)
    for i in range(width):
        bits[i] = (value >> i) & 1
    return bits


def _bits_to_int(bits: NDArray[np.int_]) -> int:
    """将低位在前 (LSB-first) 的比特数组转为无符号整数。"""
    val = 0
    for i, b in enumerate(bits):
        val |= int(b) << i
    return val


class ControlInfoType(IntEnum):
    """控制信息类型编码。"""
    A1 = 0b00001
    A2 = 0b00010
    A3 = 0b00011
    A4 = 0b00100
    A5 = 0b00101
    A6 = 0b00110
    A7 = 0b00111
    B1 = 0b10001
    B2 = 0b10010
    B3 = 0b10011
    B4 = 0b10100
    B5 = 0b10101


# =========================================================================
# A 组控制信息 (帧类型 1/2)
# =========================================================================


def _a_group_pack(info_bits: NDArray[np.int_], sync_seed: int,
                  lqi: int | None = None) -> NDArray[np.int_]:
    """A 组通用打包: 拼接 LQI(可选) + info_bits + CRC12。"""
    parts = []
    if lqi is not None:
        parts.append(_int_to_bits(lqi, 8))
    parts.append(info_bits)
    payload = np.concatenate(parts)
    return crc_attach(payload, CRC12_POLY, 12, seed=sync_seed)


def _a_group_unpack(bits: NDArray[np.int_], sync_seed: int,
                    has_lqi: bool) -> tuple[int | None, NDArray[np.int_]]:
    """A 组通用解包: 校验 CRC12, 返回 (lqi, info_bits)。"""
    if not crc_check(bits, CRC12_POLY, 12, seed=sync_seed):
        raise ValueError("CRC12 校验失败")
    payload = bits[:-12]
    lqi = None
    if has_lqi:
        lqi = _bits_to_int(payload[:8])
        payload = payload[8:]
    return lqi, payload


# ---------------------------------------------------------------------------
# A1: 广播 / 发现 / 接入
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA1:
    broadcast_type: int   # 3 bit
    packet_type: int      # 3 bit
    reserved: int         # 6 bit
    data_length: int      # 8 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.broadcast_type, 3),
            _int_to_bits(self.packet_type, 3),
            _int_to_bits(self.reserved, 6),
            _int_to_bits(self.data_length, 8),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA1]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        broadcast_type = _bits_to_int(info[off:off + 3])
        off += 3
        packet_type = _bits_to_int(info[off:off + 3])
        off += 3
        reserved = _bits_to_int(info[off:off + 6])
        off += 6
        data_length = _bits_to_int(info[off:off + 8])
        return lqi, cls(broadcast_type, packet_type, reserved, data_length)


# ---------------------------------------------------------------------------
# A2: 异步单播
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA2:
    packet_type: int     # 2 bit
    empty_packet: int    # 1 bit
    tx_sn: int           # 1 bit
    rx_sn: int           # 1 bit
    flow_ctrl: int       # 1 bit
    sys_mgmt_rx: int     # 1 bit
    reserved: int        # 2 bit
    data_length: int     # 11 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_type, 2),
            _int_to_bits(self.empty_packet, 1),
            _int_to_bits(self.tx_sn, 1),
            _int_to_bits(self.rx_sn, 1),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.sys_mgmt_rx, 1),
            _int_to_bits(self.reserved, 2),
            _int_to_bits(self.data_length, 11),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA2]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        packet_type = _bits_to_int(info[off:off + 2])
        off += 2
        empty_packet = _bits_to_int(info[off:off + 1])
        off += 1
        tx_sn = _bits_to_int(info[off:off + 1])
        off += 1
        rx_sn = _bits_to_int(info[off:off + 1])
        off += 1
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        sys_mgmt_rx = _bits_to_int(info[off:off + 1])
        off += 1
        reserved = _bits_to_int(info[off:off + 2])
        off += 2
        data_length = _bits_to_int(info[off:off + 11])
        return lqi, cls(packet_type, empty_packet, tx_sn, rx_sn,
                        flow_ctrl, sys_mgmt_rx, reserved, data_length)


# ---------------------------------------------------------------------------
# A3: 同步单播
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA3:
    packet_type: int     # 2 bit
    empty_packet: int    # 1 bit
    tx_sn: int           # 5 bit
    rx_sn: int           # 5 bit
    flow_ctrl: int       # 1 bit
    async_sched: int     # 1 bit
    reserved: int        # 2 bit
    data_length: int     # 11 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_type, 2),
            _int_to_bits(self.empty_packet, 1),
            _int_to_bits(self.tx_sn, 5),
            _int_to_bits(self.rx_sn, 5),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.async_sched, 1),
            _int_to_bits(self.reserved, 2),
            _int_to_bits(self.data_length, 11),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA3]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        packet_type = _bits_to_int(info[off:off + 2])
        off += 2
        empty_packet = _bits_to_int(info[off:off + 1])
        off += 1
        tx_sn = _bits_to_int(info[off:off + 5])
        off += 5
        rx_sn = _bits_to_int(info[off:off + 5])
        off += 5
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        async_sched = _bits_to_int(info[off:off + 1])
        off += 1
        reserved = _bits_to_int(info[off:off + 2])
        off += 2
        data_length = _bits_to_int(info[off:off + 11])
        return lqi, cls(packet_type, empty_packet, tx_sn, rx_sn,
                        flow_ctrl, async_sched, reserved, data_length)


# ---------------------------------------------------------------------------
# A4: 组播组长 (异步 / 同步)
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA4:
    packet_type: int     # 2 bit
    empty_packet: int    # 1 bit
    tx_sn: int           # 1 bit
    rx_sn: int           # 8 bit
    flow_ctrl: int       # 1 bit
    async_sched: int     # 1 bit
    reserved: int        # 3 bit
    data_length: int     # 11 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_type, 2),
            _int_to_bits(self.empty_packet, 1),
            _int_to_bits(self.tx_sn, 1),
            _int_to_bits(self.rx_sn, 8),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.async_sched, 1),
            _int_to_bits(self.reserved, 3),
            _int_to_bits(self.data_length, 11),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA4]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        packet_type = _bits_to_int(info[off:off + 2])
        off += 2
        empty_packet = _bits_to_int(info[off:off + 1])
        off += 1
        tx_sn = _bits_to_int(info[off:off + 1])
        off += 1
        rx_sn = _bits_to_int(info[off:off + 8])
        off += 8
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        async_sched = _bits_to_int(info[off:off + 1])
        off += 1
        reserved = _bits_to_int(info[off:off + 3])
        off += 3
        data_length = _bits_to_int(info[off:off + 11])
        return lqi, cls(packet_type, empty_packet, tx_sn, rx_sn,
                        flow_ctrl, async_sched, reserved, data_length)


# ---------------------------------------------------------------------------
# A5: 组播组员
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA5:
    packet_type: int     # 2 bit
    empty_packet: int    # 1 bit
    tx_sn: int           # 1 bit
    rx_sn: int           # 1 bit
    flow_ctrl: int       # 1 bit
    async_sched: int     # 1 bit
    reserved: int        # 2 bit
    data_length: int     # 11 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_type, 2),
            _int_to_bits(self.empty_packet, 1),
            _int_to_bits(self.tx_sn, 1),
            _int_to_bits(self.rx_sn, 1),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.async_sched, 1),
            _int_to_bits(self.reserved, 2),
            _int_to_bits(self.data_length, 11),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA5]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        packet_type = _bits_to_int(info[off:off + 2])
        off += 2
        empty_packet = _bits_to_int(info[off:off + 1])
        off += 1
        tx_sn = _bits_to_int(info[off:off + 1])
        off += 1
        rx_sn = _bits_to_int(info[off:off + 1])
        off += 1
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        async_sched = _bits_to_int(info[off:off + 1])
        off += 1
        reserved = _bits_to_int(info[off:off + 2])
        off += 2
        data_length = _bits_to_int(info[off:off + 11])
        return lqi, cls(packet_type, empty_packet, tx_sn, rx_sn,
                        flow_ctrl, async_sched, reserved, data_length)


# ---------------------------------------------------------------------------
# A6: 同步广播
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA6:
    packet_type: int        # 2 bit
    packet_sn: int          # 5 bit
    packet_group: int       # 3 bit
    end_indicator: int      # 1 bit
    sys_mgmt_rx: int        # 1 bit
    reserved: int           # 5 bit
    data_length: int        # 11 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_type, 2),
            _int_to_bits(self.packet_sn, 5),
            _int_to_bits(self.packet_group, 3),
            _int_to_bits(self.end_indicator, 1),
            _int_to_bits(self.sys_mgmt_rx, 1),
            _int_to_bits(self.reserved, 5),
            _int_to_bits(self.data_length, 11),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA6]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        packet_type = _bits_to_int(info[off:off + 2])
        off += 2
        packet_sn = _bits_to_int(info[off:off + 5])
        off += 5
        packet_group = _bits_to_int(info[off:off + 3])
        off += 3
        end_indicator = _bits_to_int(info[off:off + 1])
        off += 1
        sys_mgmt_rx = _bits_to_int(info[off:off + 1])
        off += 1
        reserved = _bits_to_int(info[off:off + 5])
        off += 5
        data_length = _bits_to_int(info[off:off + 11])
        return lqi, cls(packet_type, packet_sn, packet_group,
                        end_indicator, sys_mgmt_rx, reserved, data_length)


# ---------------------------------------------------------------------------
# A7: 系统管理帧
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoA7:
    packet_sn: int       # 5 bit
    packet_group: int    # 3 bit
    reserved: int        # 9 bit
    data_length: int     # 11 bit

    def pack(self, sync_seed: int, lqi: int | None = None) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_sn, 5),
            _int_to_bits(self.packet_group, 3),
            _int_to_bits(self.reserved, 9),
            _int_to_bits(self.data_length, 11),
        ])
        return _a_group_pack(info, sync_seed, lqi)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], sync_seed: int,
               has_lqi: bool = False) -> tuple[int | None, ControlInfoA7]:
        lqi, info = _a_group_unpack(bits, sync_seed, has_lqi)
        off = 0
        packet_sn = _bits_to_int(info[off:off + 5])
        off += 5
        packet_group = _bits_to_int(info[off:off + 3])
        off += 3
        reserved = _bits_to_int(info[off:off + 9])
        off += 9
        data_length = _bits_to_int(info[off:off + 11])
        return lqi, cls(packet_sn, packet_group, reserved, data_length)


# =========================================================================
# B 组控制信息 (帧类型 3/4)
# =========================================================================

_CRC24B_SEED = 0x555555


def _llid_mask(llid: int) -> NDArray[np.int_]:
    """将 LLID 扩展为 24 位掩码用于 CRC 异或。"""
    return _int_to_bits(llid & 0xFFFFFF, 24)


def _b_group_pack(info_bits: NDArray[np.int_], llid: int) -> NDArray[np.int_]:
    """B 组通用打包: info_bits + CRC24B (种子 0x555555, 异或 LLID)。"""
    mask = _llid_mask(llid)
    return crc_attach(info_bits, CRC24B_POLY, 24, seed=_CRC24B_SEED, mask=mask)


def _b_group_check(bits: NDArray[np.int_], llid: int) -> NDArray[np.int_]:
    """B 组通用解包: 校验 CRC24B, 返回 info_bits。"""
    mask = _llid_mask(llid)
    if not crc_check(bits, CRC24B_POLY, 24, seed=_CRC24B_SEED, mask=mask):
        raise ValueError("CRC24B 校验失败")
    return bits[:-24]


# ---------------------------------------------------------------------------
# B1: 单播 / 组播 PSK
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoB1:
    frame_format: int     # 1 bit
    harq_feedback: int    # 8 bit
    packet_sn: int        # 1 bit
    mcs: int              # 4 bit
    data_length: int      # 11 bit
    flow_ctrl: int        # 1 bit
    upper_link: int       # 1 bit

    def pack(self, llid: int) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.frame_format, 1),
            _int_to_bits(self.harq_feedback, 8),
            _int_to_bits(self.packet_sn, 1),
            _int_to_bits(self.mcs, 4),
            _int_to_bits(self.data_length, 11),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.upper_link, 1),
        ])
        return _b_group_pack(info, llid)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], llid: int) -> ControlInfoB1:
        info = _b_group_check(bits, llid)
        off = 0
        frame_format = _bits_to_int(info[off:off + 1])
        off += 1
        harq_feedback = _bits_to_int(info[off:off + 8])
        off += 8
        packet_sn = _bits_to_int(info[off:off + 1])
        off += 1
        mcs = _bits_to_int(info[off:off + 4])
        off += 4
        data_length = _bits_to_int(info[off:off + 11])
        off += 11
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        upper_link = _bits_to_int(info[off:off + 1])
        return cls(frame_format, harq_feedback, packet_sn, mcs,
                   data_length, flow_ctrl, upper_link)


# ---------------------------------------------------------------------------
# B2: 反馈组播组长
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoB2:
    harq_feedback: int    # 25 bit
    flow_ctrl: int        # 1 bit
    upper_link: int       # 1 bit

    def pack(self, llid: int) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.harq_feedback, 25),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.upper_link, 1),
        ])
        return _b_group_pack(info, llid)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], llid: int) -> ControlInfoB2:
        info = _b_group_check(bits, llid)
        off = 0
        harq_feedback = _bits_to_int(info[off:off + 25])
        off += 25
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        upper_link = _bits_to_int(info[off:off + 1])
        return cls(harq_feedback, flow_ctrl, upper_link)


# ---------------------------------------------------------------------------
# B3: 同步广播 (帧类型 3/4)
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoB3:
    packet_group: int        # 5 bit
    packet_sn: int           # 5 bit
    mcs: int                 # 4 bit
    data_length: int         # 11 bit
    flow_ctrl: int           # 1 bit
    max_sn_indicator: int    # 1 bit

    def pack(self, llid: int) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.packet_group, 5),
            _int_to_bits(self.packet_sn, 5),
            _int_to_bits(self.mcs, 4),
            _int_to_bits(self.data_length, 11),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.max_sn_indicator, 1),
        ])
        return _b_group_pack(info, llid)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], llid: int) -> ControlInfoB3:
        info = _b_group_check(bits, llid)
        off = 0
        packet_group = _bits_to_int(info[off:off + 5])
        off += 5
        packet_sn = _bits_to_int(info[off:off + 5])
        off += 5
        mcs = _bits_to_int(info[off:off + 4])
        off += 4
        data_length = _bits_to_int(info[off:off + 11])
        off += 11
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        max_sn_indicator = _bits_to_int(info[off:off + 1])
        return cls(packet_group, packet_sn, mcs, data_length,
                   flow_ctrl, max_sn_indicator)


# ---------------------------------------------------------------------------
# B4: 异步广播 (帧类型 3/4)
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoB4:
    broadcast_group_id: int   # 5 bit
    data_update: int          # 1 bit
    packet_sn: int            # 4 bit
    mcs: int                  # 4 bit
    data_length: int          # 11 bit
    flow_ctrl: int            # 1 bit
    max_sn_indicator: int     # 1 bit

    def pack(self, llid: int) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.broadcast_group_id, 5),
            _int_to_bits(self.data_update, 1),
            _int_to_bits(self.packet_sn, 4),
            _int_to_bits(self.mcs, 4),
            _int_to_bits(self.data_length, 11),
            _int_to_bits(self.flow_ctrl, 1),
            _int_to_bits(self.max_sn_indicator, 1),
        ])
        return _b_group_pack(info, llid)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], llid: int) -> ControlInfoB4:
        info = _b_group_check(bits, llid)
        off = 0
        broadcast_group_id = _bits_to_int(info[off:off + 5])
        off += 5
        data_update = _bits_to_int(info[off:off + 1])
        off += 1
        packet_sn = _bits_to_int(info[off:off + 4])
        off += 4
        mcs = _bits_to_int(info[off:off + 4])
        off += 4
        data_length = _bits_to_int(info[off:off + 11])
        off += 11
        flow_ctrl = _bits_to_int(info[off:off + 1])
        off += 1
        max_sn_indicator = _bits_to_int(info[off:off + 1])
        return cls(broadcast_group_id, data_update, packet_sn, mcs,
                   data_length, flow_ctrl, max_sn_indicator)


# ---------------------------------------------------------------------------
# B5: 基础广播 / 发现 / 接入 (帧类型 3/4)
# ---------------------------------------------------------------------------

@dataclass
class ControlInfoB5:
    reserved: int              # 7 bit
    msg_type: int              # 3 bit
    accessible: int            # 1 bit
    queryable: int             # 1 bit
    directed_content: int      # 1 bit
    undirected_content: int    # 1 bit
    data_update: int           # 1 bit
    mcs: int                   # 4 bit
    data_length: int           # 8 bit

    def pack(self, llid: int) -> NDArray[np.int_]:
        info = np.concatenate([
            _int_to_bits(self.reserved, 7),
            _int_to_bits(self.msg_type, 3),
            _int_to_bits(self.accessible, 1),
            _int_to_bits(self.queryable, 1),
            _int_to_bits(self.directed_content, 1),
            _int_to_bits(self.undirected_content, 1),
            _int_to_bits(self.data_update, 1),
            _int_to_bits(self.mcs, 4),
            _int_to_bits(self.data_length, 8),
        ])
        return _b_group_pack(info, llid)

    @classmethod
    def unpack(cls, bits: NDArray[np.int_], llid: int) -> ControlInfoB5:
        info = _b_group_check(bits, llid)
        off = 0
        reserved = _bits_to_int(info[off:off + 7])
        off += 7
        msg_type = _bits_to_int(info[off:off + 3])
        off += 3
        accessible = _bits_to_int(info[off:off + 1])
        off += 1
        queryable = _bits_to_int(info[off:off + 1])
        off += 1
        directed_content = _bits_to_int(info[off:off + 1])
        off += 1
        undirected_content = _bits_to_int(info[off:off + 1])
        off += 1
        data_update = _bits_to_int(info[off:off + 1])
        off += 1
        mcs = _bits_to_int(info[off:off + 4])
        off += 4
        data_length = _bits_to_int(info[off:off + 8])
        return cls(reserved, msg_type, accessible, queryable,
                   directed_content, undirected_content, data_update,
                   mcs, data_length)


# =========================================================================
# 帧类型 2 Polar 编码辅助
# =========================================================================

def polar_encode_control(raw_bits: NDArray[np.int_], n_coded: int = 64) -> NDArray[np.int_]:
    """对帧类型 2 的 A 组控制信息进行 Polar 编码。

    输入 40 或 48 比特 (含 LQI + CRC12), 编码到 64 比特。
    """
    from nearlink_sdr.common.polar import PolarEncoder

    k = len(raw_bits)
    encoder = PolarEncoder(n_coded, k)
    return encoder.encode(raw_bits)


def polar_decode_control(coded_bits: NDArray[np.int_], k_info: int,
                         n_coded: int = 64) -> NDArray[np.int_]:
    """对帧类型 2 的 A 组控制信息进行 Polar 解码。

    输入 64 比特, 解码到 k_info 比特 (40 或 48)。
    """
    from nearlink_sdr.common.polar import get_polar_decoder

    decoder = get_polar_decoder(n_coded, k_info)
    llr = (1 - 2 * coded_bits.astype(float)) * 10.0
    return decoder.decode(llr)
