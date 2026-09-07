"""码块分割 -- TXS-10002-2025 标准 6.9.1.2 / 6.9.1.3 节。"""


__all__ = [
    "RATE_TABLE_2",
    "segment_with_crc",
    "segment_without_crc",
]


import math

import numpy as np

from nearlink_sdr.common.crc import CRC24B_POLY, crc_calculate
from nearlink_sdr.common.polar import RATE_TABLE

# 表 24：6.9.1.2（无 CRC 分割）备用速率匹配表
RATE_TABLE_2 = {
    "5/8": {1024: 640, 512: 316, 256: 156, 128: 74, 64: 36},
    "3/4": {1024: 768, 512: 382, 256: 189, 128: 90, 64: 45},
    "7/8": {1024: 896, 512: 446, 256: 221, 128: 106, 64: 53},
}

# 表 20：分割查表
# 索引 -> (N512, N256, N128)
_SEG_TABLE_20 = [
    (0, 0, 0),  # 0
    (0, 0, 1),  # 1
    (0, 0, 2),  # 2
    (0, 1, 1),  # 3
    (0, 1, 2),  # 4
    (0, 2, 1),  # 5
    (0, 2, 2),  # 6
    (1, 1, 1),  # 7
    (1, 1, 2),  # 8
    (1, 2, 1),  # 9
    (1, 2, 2),  # 10
    (2, 1, 1),  # 11
    (2, 1, 2),  # 12
    (2, 2, 1),  # 13
    (2, 2, 2),  # 14
]

_RATE_FRACTIONS = {
    "1/4": 0.25,
    "3/8": 0.375,
    "1/2": 0.5,
    "5/8": 0.625,
    "3/4": 0.75,
    "7/8": 0.875,
}


def _get_rate_value(rate_str: str) -> float:
    if rate_str not in _RATE_FRACTIONS:
        raise ValueError(f"Unsupported rate '{rate_str}'")
    return _RATE_FRACTIONS[rate_str]


def _get_k_for_segmentation(rate_str: str, use_table24: bool = False) -> dict[int, int]:
    """获取给定码率下各码长对应的 K 值。

    6.9.1.2（无 CRC 分割）时, 若码率在表 24 中则 use_table24=True。
    6.9.1.3（含 CRC 分割）时, 始终使用表 23。
    """
    if use_table24 and rate_str in RATE_TABLE_2:
        return RATE_TABLE_2[rate_str]
    return RATE_TABLE[rate_str]


def segment_without_crc(
    bits: np.ndarray,
    rate_str: str,
    crc_len: int = 0,
) -> list[tuple[int, np.ndarray]]:
    """无逐码块 CRC 的码块分割 (标准 6.9.1.2 节)。

    适用于帧类型 2。

    :param bits: 输入比特序列 b_0..b_{B-1}, 其中 B = K + L (信息位 + CRC)。
    :param rate_str: 目标码率字符串。
    :param crc_len: L, 已追加到 bits 末尾的 CRC 长度。

    :returns: 每个分段的 (码长 N, 信息比特) 元组列表。
    """
    B = len(bits)
    R = _get_rate_value(rate_str)

    # 各码长对应的 K 值，优先取表 24（若有），否则取表 23
    k_table = _get_k_for_segmentation(rate_str, use_table24=True)
    K_1024 = k_table[1024]
    K_512 = k_table[512]
    K_256 = k_table[256]
    K_128 = k_table[128]
    K_64 = k_table[64]

    # 步骤 1：确定 N_1024
    threshold = R * (1024 + 512 + 256 + 128)
    if threshold < B:
        N_1024 = math.floor(
            (B - (512 + 256 + 128 + 8) * R) / (1024 * R)
        )
        K_m = B - N_1024 * K_1024
    else:
        N_1024 = 0
        K_m = B

    # 步骤 2：查表 20
    index = 0 if K_m <= 0 else math.floor((K_m - 1) / (128 * R))

    index = min(index, 14)  # 限制在表范围内
    N_512, N_256, N_128 = _SEG_TABLE_20[index]

    # 步骤 3：N_64
    remaining = K_m - N_512 * K_512 - N_256 * K_256 - N_128 * K_128
    N_64 = math.ceil(remaining / K_64) if remaining > 0 else 0

    # 构建输出分段
    segments = []
    pos = 0

    for _ in range(N_1024):
        seg_bits = bits[pos : pos + K_1024]
        segments.append((1024, seg_bits))
        pos += K_1024

    for _ in range(N_512):
        seg_bits = bits[pos : pos + K_512]
        segments.append((512, seg_bits))
        pos += K_512

    for _ in range(N_256):
        seg_bits = bits[pos : pos + K_256]
        segments.append((256, seg_bits))
        pos += K_256

    for _ in range(N_128):
        seg_bits = bits[pos : pos + K_128]
        segments.append((128, seg_bits))
        pos += K_128

    for i in range(N_64):
        if i < N_64 - 1:
            seg_bits = bits[pos : pos + K_64]
            pos += K_64
        else:
            # 最后一个 N_64 块取剩余比特（可能 < K_64）
            seg_bits = bits[pos:]
            pos = len(bits)
        segments.append((64, seg_bits))

    return segments


def segment_with_crc(
    bits: np.ndarray,
    rate_str: str,
) -> list[tuple[int, np.ndarray]]:
    """含逐码块 CRC24B 的码块分割 (标准 6.9.1.3 节)。

    适用于帧类型 3 和 4。

    :param bits: 输入比特序列 b_0..b_{B-1}。
    :param rate_str: 目标码率字符串。

    :returns: 每个分段的 (码长 N, 含 CRC 的信息比特) 元组列表。
        最后一个分段可能进一步做子分割。
    """
    B = len(bits)
    _get_rate_value(rate_str)  # validate

    # K_cb：码率 R 下 1024 码长的最大信息位数
    k_table = RATE_TABLE[rate_str]
    K_cb = k_table[1024]

    # 确定码块数量
    if K_cb >= B:
        C = 1
        L = 0
    else:
        L = 24
        C = math.ceil(B / (K_cb - L))

    # Generate segments
    segments = []
    pos = 0

    if C == 1:
        # 单分段，无逐块 CRC
        segments.append((1024, bits.copy()))
    else:
        for r in range(C):
            K_r = K_cb if r <= C - 2 else B - (C - 1) * (K_cb - L) + L

            # 提取本分段的信息位
            info_len = K_r - L
            seg_info = bits[pos : pos + info_len]
            pos += info_len

            # 追加 CRC24B
            crc_bits = crc_calculate(seg_info, CRC24B_POLY, 24)
            seg_with_crc = np.concatenate([seg_info, crc_bits])
            segments.append((1024, seg_with_crc))

    # 对最后一个分段按标准进行子分割
    if len(segments) > 0:
        _, last_bits = segments[-1]
        sub_segs = _subsegment_last_block(last_bits, rate_str)
        if sub_segs is not None:
            segments = segments[:-1] + sub_segs

    return segments


def _subsegment_last_block(
    bits: np.ndarray,
    rate_str: str,
) -> list[tuple[int, np.ndarray]] | None:
    """按标准 6.9.1.3 节对最后一个码块进行子分割。

    若无需子分割（K_r 在码率 R 下可放入 1024 块中）则返回 None。
    """
    K_r = len(bits)
    _get_rate_value(rate_str)  # validate
    k_table = RATE_TABLE[rate_str]
    K_1024 = k_table[1024]

    # 检查是否可直接放入 1024 块
    if K_r <= K_1024:
        # 无需子分割，按需填充
        if K_r < K_1024:
            padded = np.concatenate([
                np.zeros(K_1024 - K_r, dtype=np.int8),
                bits,
            ])
        else:
            padded = bits
        return [(1024, padded)]

    # K_r > K_1024：对所有标准码率均有 K_1024 > 1024 * R_adj，
    # 因此始终截断至 K_1024。
    return [(1024, bits[:K_1024])]
