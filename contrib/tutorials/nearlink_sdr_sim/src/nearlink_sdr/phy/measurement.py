"""位置信息测量信号 -- TXS-10002-2025 标准 6.2.4

实现 SLE 位置信息测量所需的信号生成:
- 窄带波形测量信号 1 (6.2.4.1)
- 窄带波形测量信号 2 (6.2.4.2)
- 多天线测量信号天线对排序 (6.2.4.3)
"""

from __future__ import annotations

__all__ = [
    "SecurityType",
    "antenna_pair_order_random",
    "antenna_pair_order_sequential",
    "measurement_signal_1",
    "measurement_signal_2",
]


from enum import IntEnum

import numpy as np

from nearlink_sdr.phy.sync_sequence import _secure_random_sequence

# ---------------------------------------------------------------------------
# 6.2.4.1 窄带波形测量信号 1
# ---------------------------------------------------------------------------

# 合法测量序列长度
_VALID_N_MEASUR = {16, 32, 64, 128, 256, 512, 1024, 2048}


class SecurityType(IntEnum):
    """测量信号 1 的安全类型配置。"""

    TYPE_1 = 1
    TYPE_2 = 2
    TYPE_3 = 3
    TYPE_4 = 4


def _extract_security_subseq(
    security_seq: np.ndarray,
    is_tx: bool,
) -> np.ndarray:
    """从 256 比特安全序列中提取 128 比特子序列。

    先发节点: 偶数索引比特 (128-bit)
    后发节点: 奇数索引比特 (128-bit)
    """
    if is_tx:
        return security_seq[0::2].copy()
    return security_seq[1::2].copy()


def _gen_signal1_type1(
    sub_seq: np.ndarray,
    n_measur: int,
    n_disturb_max: int,
) -> np.ndarray:
    """安全类型 1: 扰动符号取反。"""
    a = int(sub_seq[0])

    # N_disturb: 索引 1~4 的 4 比特转十进制 // n_disturb_max % n_disturb_max
    if n_disturb_max == 0:
        n_disturb = 0
    else:
        val_4bit = int(
            sub_seq[1] * 8 + sub_seq[2] * 4 + sub_seq[3] * 2 + sub_seq[4]
        )
        n_disturb = (val_4bit // n_disturb_max) % n_disturb_max

    # 计算扰动符号索引集合
    disturb_indices: set[int] = set()
    for n in range(n_disturb):
        start = n * 11 + 5
        end = start + 11
        bits_11 = sub_seq[start:end]
        val_11 = 0
        for b in bits_11:
            val_11 = (val_11 << 1) | int(b)
        idx = (val_11 // n_measur) % n_measur
        disturb_indices.add(idx)

    # 生成比特序列
    result = np.full(n_measur, a, dtype=int)
    for idx in disturb_indices:
        result[idx] = 1 - a
    return result


def _gen_signal1_type2(
    sub_seq: np.ndarray,
    n_measur: int,
    n_disturb_max: int,
) -> np.ndarray:
    """安全类型 2: 扰动符号置 1, 其他置 0。"""
    if n_disturb_max == 0:
        n_disturb = 0
    else:
        val_4bit = int(
            sub_seq[1] * 8 + sub_seq[2] * 4 + sub_seq[3] * 2 + sub_seq[4]
        )
        n_disturb = (val_4bit // n_disturb_max) % n_disturb_max

    disturb_indices: set[int] = set()
    for n in range(n_disturb):
        start = n * 11 + 5
        end = start + 11
        bits_11 = sub_seq[start:end]
        val_11 = 0
        for b in bits_11:
            val_11 = (val_11 << 1) | int(b)
        idx = (val_11 // n_measur) % n_measur
        disturb_indices.add(idx)

    result = np.zeros(n_measur, dtype=int)
    for idx in disturb_indices:
        result[idx] = 1
    return result


def measurement_signal_1(
    n_measur: int,
    security_type: SecurityType,
    seed: bytes = b"\x00" * 16,
    slot_number: int = 0,
    is_tx: bool = True,
    n_disturb_max: int = 11,
    kdf_type: int = 0,
) -> np.ndarray:
    """生成测量信号 1 的比特序列 (6.2.4.1)。

    无相位旋转 BPSK 调制, 返回比特序列。

    :param n_measur: 测量序列长度 [16, 32, 64, 128, 256, 512, 1024, 2048]
    :param security_type: 安全类型 (1~4)
    :param seed: 安全随机种子 (16 字节)
    :param slot_number: 起始调度时隙号
    :param is_tx: True=先发节点, False=后发节点
    :param n_disturb_max: 最大扰动符号数配置 [0-11]
    :param kdf_type: 0=AES-CMAC, 1=HMAC-SM3

    :returns: np.ndarray, dtype=int, 长度 n_measur
    """
    if n_measur not in _VALID_N_MEASUR:
        raise ValueError(
            f"n_measur 必须为 {sorted(_VALID_N_MEASUR)} 之一, 收到 {n_measur}"
        )

    st = SecurityType(security_type)

    if st == SecurityType.TYPE_4:
        return np.zeros(n_measur, dtype=int)

    # 安全类型 1/2/3 均需要安全序列
    time_param = slot_number * 2 + 1
    security_seq = _secure_random_sequence(seed, time_param, kdf_type)
    sub_seq = _extract_security_subseq(security_seq, is_tx)

    if st == SecurityType.TYPE_3:
        # 先发节点: 全部 = 安全序列索引 0; 后发节点: 全部 = 索引 1
        bit_val = int(sub_seq[0]) if is_tx else int(sub_seq[1])
        return np.full(n_measur, bit_val, dtype=int)

    if st == SecurityType.TYPE_1:
        return _gen_signal1_type1(sub_seq, n_measur, n_disturb_max)

    # SecurityType.TYPE_2
    return _gen_signal1_type2(sub_seq, n_measur, n_disturb_max)


# ---------------------------------------------------------------------------
# 6.2.4.2 窄带波形测量信号 2 (多音信号)
# ---------------------------------------------------------------------------

# 多音相位表 (弧度)
MULTITONE_PHASES: dict[int, dict[int, list[float]]] = {
    2: {1: [0.0, 0.0]},
    4: {
        1: [1.81, 0.0, 0.0, 1.81],
        2: [4.47, 0.0, 0.0, 4.47],
    },
    8: {
        1: [5.40, 3.84, 0.91, 0.0, 0.0, 0.91, 3.84, 5.40],
        2: [0.88, 2.45, 5.37, 0.0, 0.0, 5.37, 2.45, 0.88],
    },
}


def measurement_signal_2(
    n_tones: int,
    bandwidth_mhz: int,
    duration_us: float,
    sample_rate: float,
    phase_set: int = 1,
) -> np.ndarray:
    """生成测量信号 2 的复数基带波形 (6.2.4.2)。

    多音信号, 频率对称分布。

    :param n_tones: 音的个数 [1, 2, 4, 8]
    :param bandwidth_mhz: SLE 带宽 [1, 2, 4] MHz
    :param duration_us: 信号时长 (微秒)
    :param sample_rate: 采样率 (Hz)
    :param phase_set: 相位集合 [1, 2], N≥4 时有效

    :returns: np.ndarray, dtype=complex128, 复数基带波形
    """
    if n_tones not in {1, 2, 4, 8}:
        raise ValueError(f"n_tones 必须为 1/2/4/8, 收到 {n_tones}")
    if bandwidth_mhz not in {1, 2, 4}:
        raise ValueError(
            f"bandwidth_mhz 必须为 1/2/4, 收到 {bandwidth_mhz}"
        )

    bw = bandwidth_mhz * 1e6  # 转 Hz
    n_samples = int(duration_us * 1e-6 * sample_rate)
    t = np.arange(n_samples) / sample_rate

    if n_tones == 1:
        # 单音, 频率 0 Hz, 相位 0
        return np.ones(n_samples, dtype=complex)

    # 频率间隔
    delta_f = bw / n_tones

    # 获取相位列表
    if n_tones == 2:
        phases = MULTITONE_PHASES[2][1]
    else:
        if phase_set not in MULTITONE_PHASES[n_tones]:
            raise ValueError(
                f"N={n_tones} 不支持 phase_set={phase_set}"
            )
        phases = MULTITONE_PHASES[n_tones][phase_set]

    # 频率分布: f_i = -(N-1)/2 * Δf + i * Δf
    freqs = [-(n_tones - 1) / 2 * delta_f + i * delta_f
             for i in range(n_tones)]

    # 多音信号合成: x(t) = Σ cos(2πf_i*t + φ_i)
    signal = np.zeros(n_samples, dtype=complex)
    for i in range(n_tones):
        signal += np.exp(1j * (2 * np.pi * freqs[i] * t + phases[i]))

    return signal


# ---------------------------------------------------------------------------
# 6.2.4.3 多天线测量信号天线对排序
# ---------------------------------------------------------------------------


def antenna_pair_order_sequential(
    m: int,
    n: int,
) -> list[tuple[int, int]]:
    """顺序天线对集合: 优先变更后发节点天线 (6.2.4.3)。

    :param m: 发送天线数
    :param n: 接收天线数

    :returns: 天线对列表 [(tx_ant, rx_ant), ...]

    示例:
        m=3, n=2: [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    """
    return [(tx, rx) for tx in range(m) for rx in range(n)]


def antenna_pair_order_random(
    m: int,
    n: int,
    seed: bytes,
    slot_number: int,
    k: int,
    kdf_type: int = 0,
) -> list[tuple[int, int]]:
    """随机天线对集合 (使用第三安全序列置乱) (6.2.4.3)。

    :param m: 发送天线数
    :param n: 接收天线数
    :param seed: 安全随机种子 (16 字节)
    :param slot_number: 起始调度时隙号
    :param k: 随机数位宽
    :param kdf_type: 0=AES-CMAC, 1=HMAC-SM3

    :returns: 随机排列的天线对列表 [(tx_ant, rx_ant), ...]
    """
    # 生成顺序天线对集合作为候选
    candidates = antenna_pair_order_sequential(m, n)
    l0 = len(candidates)
    if l0 <= 1:
        return candidates

    # 第三安全序列: time_param = slot_number * 2 + 2
    time_param = slot_number * 2 + 2
    security_seq = _secure_random_sequence(seed, time_param, kdf_type)

    result: list[tuple[int, int]] = []
    for l_idx in range(l0):
        # 取 k 比特
        start = l_idx * k
        end = start + k
        bits_k = security_seq[start:end]
        val = 0
        for b in bits_k:
            val = (val << 1) | int(b)

        remaining = l0 - l_idx
        first_index = val % remaining
        result.append(candidates[first_index])
        candidates.pop(first_index)

    return result
