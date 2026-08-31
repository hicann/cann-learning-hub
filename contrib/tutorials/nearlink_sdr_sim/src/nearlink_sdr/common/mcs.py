"""编码调制表格与速率适配 -- TXS-10002-2025 标准 6.10.5 / 6.10.6

定义 MCS 索引到调制方式、编码速率的映射, 以及各编码速率下
不同码长对应的信息比特数 Kcb。
"""

from __future__ import annotations

__all__ = [
    "CODE_LENGTHS",
    "MCSEntry",
    "Modulation",
    "RateConfig",
    "get_kcb",
    "get_mcs",
]


from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import NamedTuple


class Modulation(IntEnum):
    """调制方式"""
    BPSK = 1
    QPSK = 2
    PSK8 = 3


class MCSEntry(NamedTuple):
    """MCS 表条目"""
    index: int
    modulation: Modulation
    code_rate: Fraction
    spectral_efficiency: float
    bits_per_symbol: int  # 每个调制符号承载的比特数


# 标准表22: 编码调制表格
MCS_TABLE: tuple[MCSEntry, ...] = (
    MCSEntry(0,  Modulation.BPSK, Fraction(1, 4), 0.250, 1),
    MCSEntry(1,  Modulation.BPSK, Fraction(3, 8), 0.375, 1),
    MCSEntry(2,  Modulation.QPSK, Fraction(1, 4), 0.500, 2),
    MCSEntry(3,  Modulation.QPSK, Fraction(3, 8), 0.750, 2),
    MCSEntry(4,  Modulation.QPSK, Fraction(1, 2), 1.000, 2),
    MCSEntry(5,  Modulation.QPSK, Fraction(5, 8), 1.250, 2),
    MCSEntry(6,  Modulation.QPSK, Fraction(3, 4), 1.500, 2),
    MCSEntry(7,  Modulation.QPSK, Fraction(7, 8), 1.750, 2),
    MCSEntry(8,  Modulation.QPSK, Fraction(1, 1), 2.000, 2),
    MCSEntry(9,  Modulation.PSK8, Fraction(5, 8), 1.875, 3),
    MCSEntry(10, Modulation.PSK8, Fraction(3, 4), 2.250, 3),
    MCSEntry(11, Modulation.PSK8, Fraction(7, 8), 2.625, 3),
    MCSEntry(12, Modulation.PSK8, Fraction(1, 1), 3.000, 3),
)

# MCS 索引 -> MCSEntry 快速查表
_MCS_MAP: dict[int, MCSEntry] = {e.index: e for e in MCS_TABLE}


def get_mcs(index: int) -> MCSEntry:
    """按索引获取 MCS 条目。"""
    if index not in _MCS_MAP:
        raise ValueError(f"无效 MCS 索引 {index}, 合法范围 0..12")
    return _MCS_MAP[index]


# ---------- 速率适配信息比特表 (6.10.6) ----------

# 码长列表 (降序)
CODE_LENGTHS = (1024, 512, 256, 128, 64)

# 标准表23: Kcb 第一表格
# key = 编码速率 (Fraction), value = {码长: Kcb}
RATE_ADAPT_TABLE_1: dict[Fraction, dict[int, int]] = {
    Fraction(1, 4): {1024: 256, 512: 96,  256: 48,  128: 24,  64: 12},
    Fraction(3, 8): {1024: 384, 512: 160, 256: 80,  128: 40,  64: 20},
    Fraction(1, 2): {1024: 512, 512: 224, 256: 112, 128: 56,  64: 28},
    Fraction(5, 8): {1024: 640, 512: 288, 256: 144, 128: 72,  64: 36},
    Fraction(3, 4): {1024: 768, 512: 352, 256: 176, 128: 88,  64: 44},
    Fraction(7, 8): {1024: 896, 512: 416, 256: 208, 128: 104, 64: 52},
}

# 标准表24: Kcb 第二表格 (仅适用于 5/8, 3/4, 7/8)
RATE_ADAPT_TABLE_2: dict[Fraction, dict[int, int]] = {
    Fraction(5, 8): {1024: 640, 512: 316, 256: 156, 128: 74,  64: 36},
    Fraction(3, 4): {1024: 768, 512: 382, 256: 189, 128: 90,  64: 45},
    Fraction(7, 8): {1024: 896, 512: 446, 256: 221, 128: 106, 64: 53},
}


def get_kcb(code_rate: Fraction, code_length: int, *, table: int = 1) -> int:
    """查询速率适配信息比特数 Kcb。

    :param code_rate: 编码速率, 如 Fraction(1, 4)。
    :type code_rate: Fraction
    :param code_length: 码长, 必须为 64, 128, 256, 512, 1024 之一。
    :type code_length: int
    :param table: 1 = 第一表格 (表23), 2 = 第二表格 (表24)。
    :type table: int
    :returns: 信息比特数 Kcb。
    :rtype: int
    """
    if code_length not in CODE_LENGTHS:
        raise ValueError(f"码长必须为 {CODE_LENGTHS} 之一, 收到 {code_length}")

    tbl = RATE_ADAPT_TABLE_1 if table == 1 else RATE_ADAPT_TABLE_2
    if code_rate not in tbl:
        raise ValueError(f"编码速率 {code_rate} 在表{table}中不存在")
    return tbl[code_rate][code_length]


@dataclass
class RateConfig:
    """速率配置: 从 MCS 索引导出完整的编码调制参数。"""
    mcs_index: int
    modulation: Modulation
    code_rate: Fraction
    bits_per_symbol: int
    spectral_efficiency: float

    @classmethod
    def from_mcs(cls, index: int) -> RateConfig:
        entry = get_mcs(index)
        return cls(
            mcs_index=entry.index,
            modulation=entry.modulation,
            code_rate=entry.code_rate,
            bits_per_symbol=entry.bits_per_symbol,
            spectral_efficiency=entry.spectral_efficiency,
        )

    def kcb(self, code_length: int, *, table: int = 1) -> int:
        """查询当前编码速率下指定码长的信息比特数。"""
        return get_kcb(self.code_rate, code_length, table=table)
