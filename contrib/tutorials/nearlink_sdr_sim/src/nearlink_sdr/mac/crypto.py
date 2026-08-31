"""安全子系统加密模块 -- TXS-10002-2025 标准 9.3 / 6.10.7

实现 SLE 安全子系统的密码学功能:
- KDF 密钥派生函数 (AES-CMAC / HMAC-SM3)
- 安全随机函数 (6.10.7)
- CCM Nonce 构建 (9.3.1.3)
- AES-CCM 加解密 (9.3.1.3)
- 初始化向量计算
- 会话密钥派生 (9.3.1.1)
- 链路密钥与 DH Key 验证码密钥 (9.3.4.4)
- 确认码生成 (9.3.4.3)
- DH Key 验证码 (9.3.4.5)
- 6位数字比较码 (9.3.4.6)
- 混淆算法 (9.3.4.7)
- 组播密钥管理 (9.3.2)
- 隐私管理 (9.4)
"""

from __future__ import annotations

__all__ = [
    "AuthMethod",
    "KdfType",
    "aes_ccm_decrypt",
    "aes_ccm_encrypt",
    "aes_cmac",
    "build_ccm_nonce_async",
    "build_ccm_nonce_other",
    "compute_iv",
    "derive_dh_verify_key",
    "derive_group_session_key",
    "derive_kg",
    "derive_link_key",
    "derive_session_key",
    "generate_confirm_code",
    "generate_dh_verify_code",
    "generate_group_key",
    "generate_numeric_code",
    "generate_resolvable_address",
    "hmac_sm3",
    "kdf",
    "obfuscate",
    "resolve_address",
    "secure_random_256",
]


from enum import IntEnum

from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from cryptography.hazmat.primitives.ciphers.algorithms import AES128
from cryptography.hazmat.primitives.cmac import CMAC

# ---------------------------------------------------------------------------
# KDF 枚举与基础函数
# ---------------------------------------------------------------------------


class KdfType(IntEnum):
    """密钥派生函数类型"""

    AES_CMAC = 0
    HMAC_SM3 = 1


class AuthMethod(IntEnum):
    """鉴权方式 (9.3.4.3)"""

    NUMERIC_COMPARISON = 0
    PASSKEY_ENTRY = 1
    PASSWORD_VERIFY = 2
    OOB = 3
    PSK = 4
    NO_INPUT = 5


def aes_cmac(key: bytes, msg: bytes) -> bytes:
    """AES-CMAC, 输入 128-bit key, 输出 128-bit MAC。"""
    c = CMAC(AES128(key))
    c.update(msg)
    return c.finalize()


def hmac_sm3(key: bytes, msg: bytes) -> bytes:
    """HMAC-SM3, 输出 256-bit 取低 128-bit。"""
    try:
        from gmssl import sm3 as _sm3
    except ImportError:
        raise NotImplementedError(
            "需要安装 gmssl: pip install gmssl"
        ) from None

    # gmssl sm3_hash 接受 bytes list
    # HMAC: H(K ^ opad || H(K ^ ipad || msg))
    block_size = 64
    if len(key) > block_size:
        k = bytes(
            _sm3.sm3_hash(list(key))
            if isinstance(_sm3.sm3_hash(list(key)), (list, bytearray))
            else bytes.fromhex(_sm3.sm3_hash(list(key)))
        )
    else:
        k = key
    k = k.ljust(block_size, b"\x00")

    ipad = bytes(b ^ 0x36 for b in k)
    opad = bytes(b ^ 0x5C for b in k)

    inner_hash_hex = _sm3.sm3_hash(list(ipad + msg))
    inner_hash = (
        bytes(inner_hash_hex)
        if isinstance(inner_hash_hex, (bytes, bytearray))
        else bytes.fromhex(inner_hash_hex)
    )
    outer_hash_hex = _sm3.sm3_hash(list(opad + inner_hash))
    digest = (
        bytes(outer_hash_hex)
        if isinstance(outer_hash_hex, (bytes, bytearray))
        else bytes.fromhex(outer_hash_hex)
    )
    # 取低 128-bit (后16字节)
    return digest[16:]


def kdf(kdf_type: KdfType, key: bytes, msg: bytes) -> bytes:
    """根据 kdf_type 选择 AES-CMAC 或 HMAC-SM3。"""
    if kdf_type == KdfType.AES_CMAC:
        return aes_cmac(key, msg)
    return hmac_sm3(key, msg)


def secure_random_256(
    seed: bytes,
    time_param: int,
    kdf_type: KdfType = KdfType.AES_CMAC,
) -> bytes:
    """安全随机函数 (6.10.7), 生成256比特 (32字节) 安全序列。

    对 KDF 调用两次 (分别使用 time_param 和 time_param+1 作为消息),
    拼接两个 128-bit 输出得到 256-bit 安全序列。

    :param seed: 128比特安全随机种子 (16字节)
    :param time_param: 32比特时间参数 (如调度时隙号*2)
    :param kdf_type: KDF 类型

    :returns: 32 字节安全序列
    """
    m1 = time_param.to_bytes(4, "big")
    m2 = ((time_param + 1) & 0xFFFFFFFF).to_bytes(4, "big")
    return kdf(kdf_type, seed, m1) + kdf(kdf_type, seed, m2)


# ---------------------------------------------------------------------------
# CCM Nonce 构建 (9.3.1.3)
# ---------------------------------------------------------------------------


def build_ccm_nonce_async(
    payload_count: int,
    direction: int,
    iv_base: bytes,
    data_length: int,
    flag: int = 0x49,
) -> bytes:
    """异步/同步单播组播链路的 CCM nonce (16 bytes)。

    布局: flag(1B) | payload_count[38:0] (39b) | direction (1b)
          | IV[63:0] (64b) | data_len_hi(1B) | data_len_lo(1B)
    """
    # 拼接 39-bit payload_count + 1-bit direction = 40 bits = 5 bytes
    combined_40 = ((payload_count & 0x7FFFFFFFFF) << 1) | (direction & 1)
    nonce_part = combined_40.to_bytes(5, "big") + iv_base[:8]  # 13 bytes
    return (
        flag.to_bytes(1, "big")
        + nonce_part
        + data_length.to_bytes(2, "big")
    )


def build_ccm_nonce_other(
    system_slot_seq: int,
    day_count: int,
    iv_base: bytes,
    data_length: int,
    flag: int = 0x49,
) -> bytes:
    """其他链路的 CCM nonce (16 bytes)。

    布局: flag(1B) | system_slot_seq[29:0] (30b) | day_count[9:0] (10b)
          | IV[63:0] (64b) | data_len_hi(1B) | data_len_lo(1B)
    """
    # 30-bit slot_seq + 10-bit day_count = 40 bits = 5 bytes
    combined_40 = (
        ((system_slot_seq & 0x3FFFFFFF) << 10)
        | (day_count & 0x3FF)
    )
    nonce_part = combined_40.to_bytes(5, "big") + iv_base[:8]  # 13 bytes
    return (
        flag.to_bytes(1, "big")
        + nonce_part
        + data_length.to_bytes(2, "big")
    )


# ---------------------------------------------------------------------------
# AES-CCM 加解密 (9.3.1.3)
# ---------------------------------------------------------------------------


def aes_ccm_encrypt(
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    associated_data: bytes = b"",
    mic_len: int = 4,
) -> tuple[bytes, bytes]:
    """AES-CCM 加密, 返回 (密文, MIC)。"""
    aesccm = AESCCM(key, tag_length=mic_len)
    ct_and_tag = aesccm.encrypt(
        nonce, plaintext, associated_data or None
    )
    ciphertext = ct_and_tag[:-mic_len]
    mic = ct_and_tag[-mic_len:]
    return ciphertext, mic


def aes_ccm_decrypt(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    mic: bytes,
    associated_data: bytes = b"",
    mic_len: int = 4,
) -> bytes:
    """AES-CCM 解密, 验证失败抛出 InvalidTag。"""
    aesccm = AESCCM(key, tag_length=mic_len)
    return aesccm.decrypt(
        nonce,
        ciphertext + mic,
        associated_data or None,
    )


# ---------------------------------------------------------------------------
# 初始化向量计算
# ---------------------------------------------------------------------------


def compute_iv(
    iv_base: bytes,
    sync_or_link_id: int,
    frame_type: int,
) -> bytes:
    """计算最终 IV (8 bytes)。

    FT1/FT2: iv_base 低32位 XOR sync_seq 低32位
    FT3/FT4: iv_base 低24位 XOR link_id 24位
    """
    iv_int = int.from_bytes(iv_base[:8], "big")
    if frame_type in (1, 2):
        mask = sync_or_link_id & 0xFFFFFFFF
        iv_int ^= mask
    else:
        mask = sync_or_link_id & 0xFFFFFF
        iv_int ^= mask
    return iv_int.to_bytes(8, "big")


# ---------------------------------------------------------------------------
# 会话密钥派生 (9.3.1.1)
# ---------------------------------------------------------------------------


def derive_session_key(
    kdf_type: KdfType,
    link_key: bytes,
    g_diversifier: bytes,
    t_diversifier: bytes,
    use_authenticated: bool = True,
) -> tuple[bytes, bytes | None]:
    """派生会话密钥。

    认证加密: SK = KDF(link_key, g_div || t_div), 返回 (SK, None)
    分离算法: EnK / InK 分别派生, 返回 (EnK, InK)
    """
    diversifier = g_diversifier + t_diversifier
    if use_authenticated:
        sk = kdf(kdf_type, link_key, diversifier)
        return sk, None
    enk = kdf(kdf_type, link_key, b"enk" + diversifier)
    ink = kdf(kdf_type, link_key, b"ink" + diversifier)
    return enk, ink


# ---------------------------------------------------------------------------
# 链路密钥与 DH Key 验证码密钥 (9.3.4.4)
# ---------------------------------------------------------------------------


def derive_link_key(
    kdf_type: KdfType,
    dh_key: bytes,
    ra: bytes,
    rb: bytes,
    g_addr: bytes,
    t_addr: bytes,
) -> bytes:
    """Link Key = KDF(dh_key_low128, "lk" || Ra || Rb || G_addr || T_addr)。"""
    key = dh_key[-16:]  # 低 128 bit
    msg = b"lk" + ra + rb + g_addr + t_addr
    return kdf(kdf_type, key, msg)


def derive_dh_verify_key(
    kdf_type: KdfType,
    dh_key: bytes,
    ra: bytes,
    rb: bytes,
    g_addr: bytes,
    t_addr: bytes,
) -> bytes:
    """DH Key 验证码密钥 = KDF(dh_key_low128, "dk" || ...)。"""
    key = dh_key[-16:]
    msg = b"dk" + ra + rb + g_addr + t_addr
    return kdf(kdf_type, key, msg)


# ---------------------------------------------------------------------------
# 确认码生成 (9.3.4.3)
# ---------------------------------------------------------------------------


def generate_confirm_code(
    kdf_type: KdfType,
    auth_method: AuthMethod,
    random_value: bytes,
    g_pubkey: bytes,
    t_pubkey: bytes,
    obfuscated: bytes = b"",
    is_g_node: bool = True,
) -> bytes:
    """生成确认码 (16 bytes)。

    random_value: Ra (G节点) 或 Rb (T节点), PSK鉴权时为 PSK
    obfuscated: 通行码/口令鉴权的混淆码, PSK鉴权的对端随机数
    """
    if auth_method == AuthMethod.NUMERIC_COMPARISON:
        msg = g_pubkey + t_pubkey
    elif auth_method in (
        AuthMethod.PASSKEY_ENTRY,
        AuthMethod.PASSWORD_VERIFY,
    ):
        msg = g_pubkey + t_pubkey + obfuscated
    elif auth_method == AuthMethod.OOB:
        msg = g_pubkey if is_g_node else t_pubkey
    elif auth_method == AuthMethod.PSK:
        msg = g_pubkey + t_pubkey + obfuscated
    else:
        msg = g_pubkey + t_pubkey
    return kdf(kdf_type, random_value, msg)


# ---------------------------------------------------------------------------
# DH Key 验证码 (9.3.4.5)
# ---------------------------------------------------------------------------


def generate_dh_verify_code(
    kdf_type: KdfType,
    verify_key: bytes,
    random_value: bytes,
    salt: bytes,
    g_io_cap: int,
    t_io_cap: int,
    auth_method: int,
    crypto_alg: int,
    g_psk_ind: int,
    t_psk_ind: int,
    g_addr: bytes,
    t_addr: bytes,
) -> bytes:
    """生成 DH Key 验证码 (16 bytes)。"""
    msg = (
        random_value
        + salt
        + g_io_cap.to_bytes(1, "big")
        + t_io_cap.to_bytes(1, "big")
        + auth_method.to_bytes(1, "big")
        + crypto_alg.to_bytes(1, "big")
        + g_psk_ind.to_bytes(1, "big")
        + t_psk_ind.to_bytes(1, "big")
        + g_addr
        + t_addr
    )
    return kdf(kdf_type, verify_key, msg)


# ---------------------------------------------------------------------------
# 6位数字比较码 (9.3.4.6)
# ---------------------------------------------------------------------------


def generate_numeric_code(
    kdf_type: KdfType,
    g_pubkey_x: bytes,
    t_pubkey: bytes,
    ra: bytes,
    rb: bytes,
) -> int:
    """生成6位数字比较码 (0-999999)。

    K = G节点公钥X 低128比特 (g_pubkey_x 后16字节)
    M = G节点公钥X 高128比特 || T节点公钥 || Ra || Rb
    """
    key = g_pubkey_x[-16:]  # 低 128 比特
    msg = g_pubkey_x[:16] + t_pubkey + ra + rb  # 高 128 比特 + T公钥 + Ra + Rb
    mac = kdf(kdf_type, key, msg)
    return int.from_bytes(mac, "big") % 1_000_000


# ---------------------------------------------------------------------------
# 混淆算法 (9.3.4.7)
# ---------------------------------------------------------------------------


def obfuscate(
    kdf_type: KdfType,
    g_pubkey_x: bytes,
    value: bytes,
) -> bytes:
    """混淆算法, 返回 16 bytes 混淆码。

    K = G节点公钥X 低128比特 (g_pubkey_x 后16字节)
    输出 = KDF(K, value)
    """
    return kdf(kdf_type, g_pubkey_x[-16:], value)


# ---------------------------------------------------------------------------
# 组播密钥管理 (9.3.2)
# ---------------------------------------------------------------------------


def generate_group_key(
    kdf_type: KdfType,
    rand1: bytes,
    rand2: bytes,
) -> bytes:
    """GK = KDF(RAND1, RAND2)。"""
    return kdf(kdf_type, rand1, rand2)


def derive_group_session_key(
    kdf_type: KdfType,
    gk: bytes,
    use_authenticated: bool = True,
) -> tuple[bytes, bytes | None]:
    """派生组播会话密钥。

    认证加密: GSK = KDF(GK, "gsk"), 返回 (GSK, None)
    分离算法: GEnK / GInK 分别派生, 返回 (GEnK, GInK)
    """
    if use_authenticated:
        return kdf(kdf_type, gk, b"gsk"), None
    genk = kdf(kdf_type, gk, b"genk")
    gink = kdf(kdf_type, gk, b"gink")
    return genk, gink


def derive_kg(
    kdf_type: KdfType,
    link_key: bytes,
    rand: bytes,
) -> bytes:
    """Kg = KDF(link_key, rand)。"""
    return kdf(kdf_type, link_key, rand)


# ---------------------------------------------------------------------------
# 隐私管理 (9.4)
# ---------------------------------------------------------------------------


def generate_resolvable_address(
    kdf_type: KdfType,
    irk: bytes,
    rand_part: int,
) -> int:
    """生成可解析随机标识 (48 bits)。

    hash_part = KDF(IRK, rand_part) mod 2^16

    :returns: 高16位标记 || 中16位随机 || 低16位hash
    """
    msg = rand_part.to_bytes(2, "big")
    mac = kdf(kdf_type, irk, msg)
    hash_part = int.from_bytes(mac, "big") & 0xFFFF
    # 高16位: 可解析随机标识固定标记 (01开头)
    marker = 0x4000 | ((rand_part >> 2) & 0x3FFF)
    return (marker << 32) | (rand_part << 16) | hash_part


def resolve_address(
    kdf_type: KdfType,
    irk: bytes,
    resolvable_addr: int,
) -> bool:
    """验证可解析随机标识。"""
    rand_part = (resolvable_addr >> 16) & 0xFFFF
    hash_expected = resolvable_addr & 0xFFFF
    msg = rand_part.to_bytes(2, "big")
    mac = kdf(kdf_type, irk, msg)
    hash_actual = int.from_bytes(mac, "big") & 0xFFFF
    return hash_actual == hash_expected
