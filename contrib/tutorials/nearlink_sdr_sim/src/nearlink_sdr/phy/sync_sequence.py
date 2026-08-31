
__all__ = [
    "SYNC1_BROADCAST",
    "SYNC2_BROADCAST",
    "sync_signal_1",
    "sync_signal_1_validate",
    "sync_signal_2",
    "sync_signal_3",
    "sync_signal_4",
    "sync_signal_5",
    "sync_signal_6",
]


import numpy as np

from nearlink_sdr.common.bch import bch_31_26_encode, bch_63_24_encode
from nearlink_sdr.common.m_sequence import generate_m_sequence, m31_sequence, m63_sequence

# TXS-10002-2025 6.2.3 同步序列/同步信号

# ---- 广播帧固定同步序列 ----

# 同步信号1: 0x5A2BDA62, LSB first
SYNC1_BROADCAST = np.array(
    [
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,  # 0x62
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,  # 0xDA
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,  # 0x2B
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,  # 0x5A
    ],
    dtype=int,
)

# 同步信号2: 0x7DE7585C6D226540, LSB first
SYNC2_BROADCAST = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,  # 0x40
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,  # 0x65
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,  # 0x22
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,  # 0x6D
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,  # 0x5C
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,  # 0x58
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,  # 0xE7
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,  # 0x7D
    ],
    dtype=int,
)

# 同步信号1 m序列参数: 5阶LFSR, 初始值全1
# 反馈多项式 x^5 + x^3 + 1 (M31 index 1, 见标准图5)
_SYNC1_M_TAPS = 0b101001
_SYNC1_M_INIT = 0b11111

# 同步信号2 m序列参数: 6阶LFSR, 初始值全1
# 反馈多项式 x^6 + x^5 + 1 (标准图5, M63 index 1)
_SYNC2_M_TAPS = 0b1100001
_SYNC2_M_INIT = 0b111111


def _hex_to_bits_lsb(hex_val: int, n_bits: int) -> np.ndarray:
    """将整数转换为LSB优先的比特数组。"""
    bits = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        bits[i] = (hex_val >> i) & 1
    return bits


def sync_signal_1(link_id_24: int | None = None) -> np.ndarray:
    """生成同步信号1的32比特序列。

    标准6.2.3.1:
    - 广播帧: 固定序列0x5A2BDA62
    - 其他帧: 24位逻辑链路标识 → 补"10" → BCH(31,26) → m31异或 → 补"0"

    :param link_id_24: 24位逻辑链路标识，None表示广播帧

    :returns: 32比特同步序列
    """
    if link_id_24 is None:
        return SYNC1_BROADCAST.copy()

    # 24位逻辑链路标识 → LSB优先比特数组
    a = _hex_to_bits_lsb(link_id_24, 24)

    # 第一步: 前面补"10" → Ã(d) = 1 + d^2*A(d)
    a_tilde = np.zeros(26, dtype=int)
    a_tilde[0] = 1  # 常数项1
    a_tilde[1] = 0  # d^1项为0
    a_tilde[2:26] = a  # d^2*A(d)

    # 第二步: BCH(31,26)编码
    codeword_31 = bch_31_26_encode(a_tilde)

    # 第三步: 与31比特m序列异或
    m_seq = generate_m_sequence(5, _SYNC1_M_TAPS, _SYNC1_M_INIT, 31)
    scrambled = codeword_31 ^ m_seq

    # 第四步: 末尾补"0"
    sync_32 = np.zeros(32, dtype=int)
    sync_32[:31] = scrambled
    sync_32[31] = 0

    return sync_32


def sync_signal_2(link_id_24: int | None = None) -> np.ndarray:
    """生成同步信号2的64比特序列。

    标准6.2.3.2:
    - 广播帧: 固定序列0x7DE7585C6D226540
    - 其他帧: 24位逻辑链路标识 → BCH(63,24) → m63异或 → 补"0"

    :param link_id_24: 24位逻辑链路标识，None表示广播帧

    :returns: 64比特同步序列
    """
    if link_id_24 is None:
        return SYNC2_BROADCAST.copy()

    # 24位逻辑链路标识 → LSB优先比特数组
    a = _hex_to_bits_lsb(link_id_24, 24)

    # 第一步: BCH(63,24)编码
    codeword_63 = bch_63_24_encode(a)

    # 第二步: 与63比特m序列异或
    m_seq = generate_m_sequence(6, _SYNC2_M_TAPS, _SYNC2_M_INIT, 63)
    scrambled = codeword_63 ^ m_seq

    # 第三步: 末尾补"0"
    sync_64 = np.zeros(64, dtype=int)
    sync_64[:63] = scrambled
    sync_64[63] = 0

    return sync_64


def sync_signal_3(m_seq_index: int = 0) -> np.ndarray:
    """生成同步信号3的62比特序列。

    标准6.2.3.3: 两个相同的31长m序列串联。
    经BPSK调制后产生62个符号。

    :param m_seq_index: m31序列编号 0~5，基础广播信道为0

    :returns: 62比特序列
    """
    m31 = m31_sequence(m_seq_index, 31)
    return np.concatenate([m31, m31])


def sync_signal_4(m_seq_index: int = 0) -> np.ndarray:
    """生成同步信号4的126比特序列。

    标准6.2.3.4: 两个相同的63长m序列串联。
    经BPSK调制后产生126个符号。

    :param m_seq_index: m63序列编号 0~5，基础广播信道为0

    :returns: 126比特序列
    """
    m63 = m63_sequence(m_seq_index, 63)
    return np.concatenate([m63, m63])


def sync_signal_1_validate(sync_32: np.ndarray) -> bool:
    """验证32比特同步序列是否满足标准约束条件。

    标准6.2.3.1要求:
    1. 不能出现7个或以上连续的0或1
    2. 不能出现26次或以上的0/1跳转
    3. 不能和广播帧同步序列相同
    """
    # 条件1: 连续0/1检查
    max_run = 1
    current_run = 1
    for i in range(1, len(sync_32)):
        if sync_32[i] == sync_32[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run >= 7:
        return False

    # 条件2: 跳转次数检查
    transitions = np.sum(np.abs(np.diff(sync_32)))
    if transitions >= 26:
        return False

    # 条件3: 不能和广播帧相同
    return not np.array_equal(sync_32, SYNC1_BROADCAST)


# ---------------------------------------------------------------------------
# 6.10.7 安全随机函数
# ---------------------------------------------------------------------------


def _secure_random_sequence(
    seed: bytes,
    time_param: int,
    kdf_type: int = 0,
) -> np.ndarray:
    """安全随机函数 (6.10.7), 生成256比特安全序列。

    :param seed: 128比特安全随机种子 (16字节)
    :param time_param: 32比特时间参数 (如调度时隙号×2)
    :param kdf_type: 0=AES-CMAC, 1=HMAC-SM3

    :returns: 256比特数组
    """
    from nearlink_sdr.mac.crypto import KdfType, secure_random_256

    full = secure_random_256(seed, time_param, KdfType(kdf_type))

    bits = np.zeros(256, dtype=int)
    for i in range(256):
        byte_idx = i // 8
        bit_idx = i % 8
        bits[i] = (full[byte_idx] >> (7 - bit_idx)) & 1
    return bits


def _extract_sync_from_security_seq(
    security_seq: np.ndarray,
    n_sync: int,
    is_tx: bool = True,
) -> np.ndarray:
    """从安全序列中提取随机同步序列。

    先发节点: 偶数索引比特的前 n_sync 个
    后发节点: 奇数索引比特的前 n_sync 个

    :param security_seq: 256比特安全序列
    :param n_sync: 同步序列长度 (32, 64, 128)
    :param is_tx: True=先发节点, False=后发节点
    """
    if is_tx:
        return security_seq[0::2][:n_sync].copy()
    return security_seq[1::2][:n_sync].copy()


# ---------------------------------------------------------------------------
# 6.2.3.5 同步信号 5
# ---------------------------------------------------------------------------


def sync_signal_5(
    seed: bytes,
    slot_number: int,
    n_sync: int = 32,
    is_tx: bool = True,
    kdf_type: int = 0,
) -> np.ndarray:
    """生成同步信号5的随机同步序列 (GFSK调制)。

    标准6.2.3.5: 使用安全随机函数生成256比特安全序列,
    先发/后发节点分别取偶数/奇数索引比特。

    :param seed: 安全随机种子 (16字节)
    :param slot_number: 起始调度时隙号
    :param n_sync: 同步序列长度 [32, 64, 128]
    :param is_tx: True=先发节点, False=后发节点
    :param kdf_type: 0=AES-CMAC, 1=HMAC-SM3
    """
    time_param = slot_number * 2
    security_seq = _secure_random_sequence(seed, time_param, kdf_type)
    return _extract_sync_from_security_seq(security_seq, n_sync, is_tx)


# ---------------------------------------------------------------------------
# 6.2.3.6 同步信号 6
# ---------------------------------------------------------------------------


def sync_signal_6(
    seed: bytes,
    slot_number: int,
    n_sync: int = 32,
    is_tx: bool = True,
    kdf_type: int = 0,
) -> np.ndarray:
    """生成同步信号6的随机同步序列 (无相位旋转BPSK调制)。

    标准6.2.3.6: 与同步信号5生成方式相同,
    区别仅在调制方式 (BPSK vs GFSK)。

    :param seed: 安全随机种子 (16字节)
    :param slot_number: 起始调度时隙号
    :param n_sync: 同步序列长度 [32, 64, 128]
    :param is_tx: True=先发节点, False=后发节点
    :param kdf_type: 0=AES-CMAC, 1=HMAC-SM3
    """
    time_param = slot_number * 2
    security_seq = _secure_random_sequence(seed, time_param, kdf_type)
    return _extract_sync_from_security_seq(security_seq, n_sync, is_tx)
