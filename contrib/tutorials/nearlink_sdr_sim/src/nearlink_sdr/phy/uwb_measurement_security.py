"""超宽带脉冲测量安全模块 -- TXS-10002-2025 标准 9.5

实现 UWB 测量安全框架:
  - 9.5.2  KDF 算法 (Label + Context 组装 256-bit message)
  - 9.5.3  密钥派生 (SLPKey -> ctsKey / ctsValue / ctsGap)
  - 9.5.4  TGap 生成
  - 9.5.5  CTS 测量符号索引与加扰 SC 序列生成
"""

from __future__ import annotations

__all__ = [
    "LABEL_CTS_GAP_K",
    "LABEL_CTS_INIT_C",
    "LABEL_CTS_K",
    "LABEL_CTS_V",
    "LABEL_SLP",
    "CTSKeys",
    "CTSSymbolResult",
    "EncryptionAlgo",
    "UWBMeasInputContext",
    "advance_cts_v_counter",
    "compute_tgap",
    "derive_cts_keys",
    "derive_slp_key",
    "generate_cts_symbols",
    "update_cts_keys",
]


import struct
from dataclasses import dataclass, field

from nearlink_sdr.mac.crypto import KdfType, kdf

# ---------------------------------------------------------------------------
# 9.5.2 Label 定义 (表 70, 64-bit ASCII 编码)
# ---------------------------------------------------------------------------
LABEL_SLP = bytes.fromhex("0000000000534C50")       # "SLP"
LABEL_CTS_INIT_C = bytes.fromhex("637473496E697443")  # "ctsInitC"
LABEL_CTS_K = bytes.fromhex("000000006374734B")       # "ctsK"
LABEL_CTS_GAP_K = bytes.fromhex("006374734761704B")   # "ctsGapK"
LABEL_CTS_V = bytes.fromhex("0000000063747356")       # "ctsV"


def _build_kdf_message(label: bytes, context: bytes) -> bytes:
    """构建 KDF 的 256-bit Message = 预留(64) || Label(64) || Context(128)。"""
    if len(label) != 8:
        raise ValueError("Label 长度必须为 8 字节")
    if len(context) != 16:
        raise ValueError("Context 长度必须为 16 字节")
    reserved = b"\x00" * 8
    return reserved + label + context


# ---------------------------------------------------------------------------
# 9.5.3 inputContext 结构 (表 71, 128 bit)
# ---------------------------------------------------------------------------
@dataclass
class UWBMeasInputContext:
    """UWB 测量安全的 inputContext (标准表 71)。"""
    phy_channel: int = 0          # 8 bit: 物理信道号
    ranging_method: int = 1       # 2 bit: 0=单向单消息, 1=双向两消息, 2=双向三消息
    ranging_mode: int = 0         # 2 bit: 0=一对一
    code_len_duty: int = 0        # 4 bit: 码长和占空比索引
    symbol_index: int = 0         # 8 bit: 符号索引
    nmss: int = 0                 # 32 bit: 测量符号序列长度
    g_node_l2id: bytes = b"\x00" * 6  # 48 bit: G 节点 L2ID
    meas_signal_config_index: int = 0  # 8 bit: 位置测量信号配置索引
    tx_antenna_first: int = 0     # 4 bit: 先发发送天线数
    rx_antenna_first: int = 0     # 4 bit: 先发接收天线数
    tx_antenna_second: int = 0    # 4 bit: 后发发送天线数
    rx_antenna_second: int = 0    # 4 bit: 后发接收天线数

    def pack(self) -> bytes:
        """打包为 16 字节 (128 bit)。"""
        buf = bytearray(16)
        buf[0] = self.phy_channel & 0xFF
        # 字节 1: ranging_method(2) | ranging_mode(2) | code_len_duty(4)
        buf[1] = (
            (self.ranging_method & 0x03)
            | ((self.ranging_mode & 0x03) << 2)
            | ((self.code_len_duty & 0x0F) << 4)
        )
        buf[2] = self.symbol_index & 0xFF
        # nmss: 字节 3-6 (32 bit, 实际有效 24 bit)
        struct.pack_into("<I", buf, 3, self.nmss & 0xFFFFFFFF)
        # g_node_l2id: 字节 7-12 (48 bit)
        buf[7:13] = self.g_node_l2id[:6]
        buf[13] = self.meas_signal_config_index & 0x3F
        # 字节 14: tx_antenna_first(4) | rx_antenna_first(4)
        buf[14] = (self.tx_antenna_first & 0x0F) | ((self.rx_antenna_first & 0x0F) << 4)
        # 字节 15: tx_antenna_second(4) | rx_antenna_second(4)
        buf[15] = (self.tx_antenna_second & 0x0F) | ((self.rx_antenna_second & 0x0F) << 4)
        return bytes(buf)


# ---------------------------------------------------------------------------
# 9.5.3 密钥派生
# ---------------------------------------------------------------------------
@dataclass
class CTSKeys:
    """CTS 密钥组 (ctsKey, ctsValue, ctsGap)。"""
    cts_key: bytes = b""       # 16 字节
    cts_value: bytes = b""     # 16 字节
    cts_gap: bytes = b""       # 16 字节
    cts_content: bytes = b""   # 16 字节 (MSB96 || ctsContent32Bit)
    cts_content_32bit: int = 0  # ctsContent32Bit (可递增, 低 32 bit)


def derive_slp_key(
    link_key: bytes,
    kdf_type: KdfType = KdfType.AES_CMAC,
) -> bytes:
    """从 Link Key 派生 SLPKey (步骤 1)。"""
    context = b"\x00" * 16
    msg = _build_kdf_message(LABEL_SLP, context)
    return kdf(kdf_type, link_key, msg)


def derive_cts_keys(
    slp_key: bytes,
    input_context: bytes,
    kdf_type: KdfType = KdfType.AES_CMAC,
) -> CTSKeys:
    """从 SLPKey 和 inputContext 派生 CTS 密钥组 (步骤 2-11)。"""
    if len(input_context) != 16:
        raise ValueError("inputContext 长度必须为 16 字节")

    # 步骤 2-3: ctsInitContent = KDF(SLPKey, 预留 || ctsInitC || inputContext)
    msg_init = _build_kdf_message(LABEL_CTS_INIT_C, input_context)
    cts_init_content = kdf(kdf_type, slp_key, msg_init)

    # 步骤 4: ctsContent32Bit = LSB 32 bits of ctsInitContent
    cts_content_32bit = int.from_bytes(cts_init_content[:4], "little")

    # 步骤 5: ctsContent = MSB96 || ctsContent32Bit
    msb96 = cts_init_content[4:16]  # 高 96 bit (字节 4-15)
    cts_content = msb96 + cts_init_content[:4]

    return _derive_cts_from_content(
        slp_key, cts_content, cts_content_32bit, kdf_type,
    )


def update_cts_keys(
    slp_key: bytes,
    prev_keys: CTSKeys,
    kdf_type: KdfType = KdfType.AES_CMAC,
) -> CTSKeys:
    """递增 ctsContent32Bit 并重新派生 CTS 密钥 (步骤 12)。"""
    new_32bit = (prev_keys.cts_content_32bit + 1) & 0xFFFFFFFF
    msb96 = prev_keys.cts_content[0:12]
    cts_content = msb96 + new_32bit.to_bytes(4, "little")
    return _derive_cts_from_content(slp_key, cts_content, new_32bit, kdf_type)


def _derive_cts_from_content(
    slp_key: bytes,
    cts_content: bytes,
    cts_content_32bit: int,
    kdf_type: KdfType,
) -> CTSKeys:
    """从 ctsContent 派生 ctsKey / ctsValue / ctsGap (步骤 6-11)。"""
    # 步骤 6-7: ctsKey = KDF(SLPKey, 预留 || ctsK || ctsContent)
    msg_k = _build_kdf_message(LABEL_CTS_K, cts_content)
    cts_key = kdf(kdf_type, slp_key, msg_k)

    # 步骤 8-9: ctsValue = KDF(SLPKey, 预留 || ctsV || ctsContent)
    msg_v = _build_kdf_message(LABEL_CTS_V, cts_content)
    cts_value = kdf(kdf_type, slp_key, msg_v)

    # 步骤 10-11: ctsGap = KDF(SLPKey, 预留 || ctsGapK || ctsContent)
    msg_gap = _build_kdf_message(LABEL_CTS_GAP_K, cts_content)
    cts_gap = kdf(kdf_type, slp_key, msg_gap)

    return CTSKeys(
        cts_key=cts_key,
        cts_value=cts_value,
        cts_gap=cts_gap,
        cts_content=cts_content,
        cts_content_32bit=cts_content_32bit,
    )


# ---------------------------------------------------------------------------
# 9.5.4 TGap 生成
# ---------------------------------------------------------------------------
def compute_tgap(
    cts_gap: bytes,
    cts_gap_shift: int,
    t_base: int,
    code_len: int,
    delta_l: int,
) -> int:
    """计算 TGap (标准 9.5.4)。

    :param cts_gap: 128-bit ctsGap 密钥.
    :param cts_gap_shift: 当前帧对应的移位量 (首帧为 0, 后续每帧 +1).
    :param t_base: 基准长度 (Tc 单位).
    :param code_len: 码长.
    :param delta_l: deltaL 系数.

    :returns: Tgap = Tbase - Toffset (Tc 单位).
    """
    # 将 ctsGap 视为 128-bit 整数 (小端)
    gap_int = int.from_bytes(cts_gap, "little")
    # 右移 ctsGapShift 位, 截取末尾 10 bit
    raw_10bit = (gap_int >> cts_gap_shift) & 0x3FF
    # chip 变换: 高 2bit * code_len * deltaL + 低 8bit
    high_2 = (raw_10bit >> 8) & 0x03
    low_8 = raw_10bit & 0xFF
    t_offset = high_2 * code_len * delta_l + low_8
    return t_base - t_offset


# ---------------------------------------------------------------------------
# 9.5.5 CTS 测量符号索引和加扰 SC 序列
# ---------------------------------------------------------------------------

class EncryptionAlgo:
    """加密算法枚举。"""
    NONE = 0
    SM4_128 = 1
    AES_128 = 2
    ZUC_256 = 3
    AES_256 = 4


def _encrypt_block(algo: int, key: bytes, plaintext: bytes) -> bytes:
    """单块加密 (ECB 模式, 16 字节输入输出)。"""
    if algo == EncryptionAlgo.AES_128:
        from cryptography.hazmat.primitives.ciphers import Cipher
        from cryptography.hazmat.primitives.ciphers.algorithms import AES128
        from cryptography.hazmat.primitives.ciphers.modes import ECB
        cipher = Cipher(AES128(key[:16]), ECB())
        enc = cipher.encryptor()
        return enc.update(plaintext[:16]) + enc.finalize()
    if algo == EncryptionAlgo.SM4_128:
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher
            from cryptography.hazmat.primitives.ciphers.algorithms import SM4
            from cryptography.hazmat.primitives.ciphers.modes import ECB
            cipher = Cipher(SM4(key[:16]), ECB())
            enc = cipher.encryptor()
            return enc.update(plaintext[:16]) + enc.finalize()
        except ImportError:
            pass
    # 回退: AES-128
    from cryptography.hazmat.primitives.ciphers import Cipher
    from cryptography.hazmat.primitives.ciphers.algorithms import AES128
    from cryptography.hazmat.primitives.ciphers.modes import ECB
    cipher = Cipher(AES128(key[:16]), ECB())
    enc = cipher.encryptor()
    return enc.update(plaintext[:16]) + enc.finalize()


@dataclass
class CTSSymbolResult:
    """CTS 测量符号索引和加扰序列。"""
    symbol_indices: list[int] = field(default_factory=list)
    sc_values: list[int] = field(default_factory=list)  # +1 或 -1


def generate_cts_symbols(
    cts_key: bytes,
    cts_v_upper: bytes,
    cts_v_counter: int,
    n_cts: int,
    symbol_count: int = 32,
    encryption_algo: int = EncryptionAlgo.AES_128,
) -> CTSSymbolResult:
    """生成 CTS 测量符号索引和加扰 SC 序列 (标准 9.5.5)。

    :param cts_key: ctsKey (16 字节).
    :param cts_v_upper: ctsVUpper (高 96 bit = 12 字节).
    :param cts_v_counter: ctsVCounter (32 bit).
    :param n_cts: CTS 数据部分符号个数.
    :param symbol_count: 符号索引数 (16 或 32).
    :param encryption_algo: 加密算法 (默认 AES-128).

    :returns: CTSSymbolResult 包含符号索引和 SC 序列.
    """
    result = CTSSymbolResult()
    index_bits = 4 if symbol_count <= 16 else 5
    index_mask = (1 << index_bits) - 1

    for i in range(1, n_cts + 1):
        # ctsMsg = ctsVUpper || (ctsVCounter + i - 1)
        counter_val = (cts_v_counter + i - 1) & 0xFFFFFFFF
        cts_msg = cts_v_upper[:12] + counter_val.to_bytes(4, "little")
        si_block = _encrypt_block(encryption_algo, cts_key, cts_msg)
        # 取加密输出对应字节的低比特作为符号索引
        byte_idx = (i - 1) % 16
        si = si_block[byte_idx]
        sym_idx = si & index_mask
        sc = 1 if (si & 0x80) == 0 else -1
        result.symbol_indices.append(sym_idx)
        result.sc_values.append(sc)

    return result


def advance_cts_v_counter(cts_v_counter: int, n_cts: int) -> int:
    """每帧后推进 ctsVCounter (标准 9.5.5)。"""
    n = max(n_cts // 16, 1)
    return (cts_v_counter + n) & 0xFFFFFFFF
