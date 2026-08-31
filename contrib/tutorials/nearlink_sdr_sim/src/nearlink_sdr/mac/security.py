"""安全层配对与鉴权信令 -- TXS-10002-2025 标准 9.2

实现 SLE 配对和鉴权管理所需的信令消息编解码。
"""

from __future__ import annotations

__all__ = [
    "AddrType",
    "GNodeAddress",
    "GNodeConfirmCode",
    "GNodeConfirmCodeWithRandom",
    "GNodeDHKeyVerify",
    "GNodeIRK",
    "MulticastAlgorithmConfig",
    "MulticastKeyConfig",
    "PairingConfirm",
    "PairingFailure",
    "PairingInitialInfo",
    "PairingInitiate",
    "PairingRequest",
    "PairingResponse",
    "RaMessage",
    "RbMessage",
    "ResolutionAlgorithm",
    "RgMessage",
    "RtMessage",
    "TNodeAddress",
    "TNodeConfirmCode",
    "TNodeConfirmCodeWithRandom",
    "TNodeDHKeyVerify",
    "TNodeIRK",
]


import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 辅助基类: 纯 bytes 固定长度消息
# ---------------------------------------------------------------------------

@dataclass
class _BytesMessage:
    """仅包含单个 bytes 字段的消息基类。

    子类需定义 DATA_TYPE_INDEX、BYTE_LENGTH 和一个 bytes 字段。
    """

    def pack(self) -> bytes:
        # 取第一个 dataclass 字段的值
        val: bytes = getattr(self, self.__dataclass_fields__.__iter__().__next__())
        if len(val) != self.BYTE_LENGTH:
            raise ValueError(
                f"字段长度 {len(val)} != {self.BYTE_LENGTH}"
            )
        return val

    @classmethod
    def unpack(cls, data: bytes):
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(data[: cls.BYTE_LENGTH])


# ---------------------------------------------------------------------------
# 双 bytes 字段基类 (公钥 X/Y 或 确认码+随机数)
# ---------------------------------------------------------------------------

@dataclass
class _TwoBytesMessage:
    """包含两个等长 bytes 字段的消息基类。"""

    def pack(self) -> bytes:
        names = list(self.__dataclass_fields__)
        a: bytes = getattr(self, names[0])
        b: bytes = getattr(self, names[1])
        half = self.BYTE_LENGTH // 2
        if len(a) != half or len(b) != half:
            raise ValueError(f"字段长度错误: 需要各 {half} 字节")
        return a + b

    @classmethod
    def unpack(cls, data: bytes):
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        half = cls.BYTE_LENGTH // 2
        return cls(data[:half], data[half: cls.BYTE_LENGTH])


# ---------------------------------------------------------------------------
# 9.2 配对发起 (0x0133, 1 字节)
# ---------------------------------------------------------------------------

@dataclass
class PairingInitiate:
    """配对发起 (标准 9.2)。

    auth_request 字段布局 (8 bits):
      安全属性(2bit) | 防中间人攻击指示位(1bit) | 按键提示(1bit) | 保留(4bit)
    """
    auth_request: int  # 1 字节

    DATA_TYPE_INDEX: int = 0x0133
    BYTE_LENGTH: int = 1

    def pack(self) -> bytes:
        return struct.pack("B", self.auth_request & 0xFF)

    @classmethod
    def unpack(cls, data: bytes) -> PairingInitiate:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(auth_request=struct.unpack("B", data[:1])[0])


# ---------------------------------------------------------------------------
# 9.2 配对请求 (0x0134, 10 字节)
# ---------------------------------------------------------------------------

@dataclass
class PairingRequest:
    """配对请求 (标准 9.2)。

    字段:
    - io_capability:      1 字节, I/O 能力
    - oob_data_flag:      1 字节, OOB 数据标志
    - auth_request:       1 字节, 鉴权请求
    - max_key_length:     1 字节, 最大密钥长度
    - security_dist_info: 1 字节, 安全分发信息
    - crypto_capability:  4 字节, 加密能力
    - psk_indication:     1 字节, PSK 指示
    """
    io_capability: int       # 0x00-0x04
    oob_data_flag: int       # 0x00/0x01
    auth_request: int        # 1 字节
    max_key_length: int      # 默认 16
    security_dist_info: int  # 1 字节
    crypto_capability: bytes  # 4 字节
    psk_indication: int      # 0x00/0x01

    DATA_TYPE_INDEX: int = 0x0134
    BYTE_LENGTH: int = 10

    def pack(self) -> bytes:
        if len(self.crypto_capability) != 4:
            raise ValueError("crypto_capability 需要 4 字节")
        return struct.pack(
            "5B", self.io_capability, self.oob_data_flag,
            self.auth_request, self.max_key_length,
            self.security_dist_info,
        ) + self.crypto_capability + struct.pack("B", self.psk_indication)

    @classmethod
    def unpack(cls, data: bytes) -> PairingRequest:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        vals = struct.unpack("5B", data[:5])
        crypto = data[5:9]
        psk = struct.unpack("B", data[9:10])[0]
        return cls(
            io_capability=vals[0],
            oob_data_flag=vals[1],
            auth_request=vals[2],
            max_key_length=vals[3],
            security_dist_info=vals[4],
            crypto_capability=crypto,
            psk_indication=psk,
        )


# ---------------------------------------------------------------------------
# 9.2 配对响应 (0x0135, 10 字节)
# ---------------------------------------------------------------------------

@dataclass
class PairingResponse:
    """配对响应 (标准 9.2)。

    字段与 PairingRequest 完全相同。
    """
    io_capability: int
    oob_data_flag: int
    auth_request: int
    max_key_length: int
    security_dist_info: int
    crypto_capability: bytes  # 4 字节
    psk_indication: int

    DATA_TYPE_INDEX: int = 0x0135
    BYTE_LENGTH: int = 10

    def pack(self) -> bytes:
        if len(self.crypto_capability) != 4:
            raise ValueError("crypto_capability 需要 4 字节")
        return struct.pack(
            "5B", self.io_capability, self.oob_data_flag,
            self.auth_request, self.max_key_length,
            self.security_dist_info,
        ) + self.crypto_capability + struct.pack("B", self.psk_indication)

    @classmethod
    def unpack(cls, data: bytes) -> PairingResponse:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        vals = struct.unpack("5B", data[:5])
        crypto = data[5:9]
        psk = struct.unpack("B", data[9:10])[0]
        return cls(
            io_capability=vals[0],
            oob_data_flag=vals[1],
            auth_request=vals[2],
            max_key_length=vals[3],
            security_dist_info=vals[4],
            crypto_capability=crypto,
            psk_indication=psk,
        )


# ---------------------------------------------------------------------------
# 9.2 配对确认 (0x0136, 70 字节)
# ---------------------------------------------------------------------------

@dataclass
class PairingConfirm:
    """配对确认 (标准 9.2)。

    字段:
    - key_length:       1 字节, 密钥长度
    - auth_method:      1 字节, 鉴权方法
    - crypto_algorithm: 4 字节, 加密算法能力
    - g_public_key_x:  32 字节, G 节点公钥 X 分量
    - g_public_key_y:  32 字节, G 节点公钥 Y 分量
    """
    key_length: int          # 1 字节
    auth_method: int         # 0x00-0x05
    crypto_algorithm: bytes  # 4 字节
    g_public_key_x: bytes    # 32 字节
    g_public_key_y: bytes    # 32 字节

    DATA_TYPE_INDEX: int = 0x0136
    BYTE_LENGTH: int = 70

    def pack(self) -> bytes:
        if len(self.crypto_algorithm) != 4:
            raise ValueError("crypto_algorithm 需要 4 字节")
        if len(self.g_public_key_x) != 32:
            raise ValueError("g_public_key_x 需要 32 字节")
        if len(self.g_public_key_y) != 32:
            raise ValueError("g_public_key_y 需要 32 字节")
        return (
            struct.pack("2B", self.key_length, self.auth_method)
            + self.crypto_algorithm
            + self.g_public_key_x
            + self.g_public_key_y
        )

    @classmethod
    def unpack(cls, data: bytes) -> PairingConfirm:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        key_length, auth_method = struct.unpack("2B", data[:2])
        return cls(
            key_length=key_length,
            auth_method=auth_method,
            crypto_algorithm=data[2:6],
            g_public_key_x=data[6:38],
            g_public_key_y=data[38:70],
        )


# ---------------------------------------------------------------------------
# 9.2 配对初始信息 (0x0137, 64 字节)
# ---------------------------------------------------------------------------

@dataclass
class PairingInitialInfo(_TwoBytesMessage):
    """配对初始信息 (标准 9.2): T 节点公钥。"""
    t_public_key_x: bytes  # 32 字节
    t_public_key_y: bytes  # 32 字节

    DATA_TYPE_INDEX: int = 0x0137
    BYTE_LENGTH: int = 64


# ---------------------------------------------------------------------------
# 9.2 T 节点确认码 (0x0138, 16 字节)
# ---------------------------------------------------------------------------

@dataclass
class TNodeConfirmCode(_BytesMessage):
    """T 节点确认码 (标准 9.2)。"""
    confirm_code: bytes  # 16 字节

    DATA_TYPE_INDEX: int = 0x0138
    BYTE_LENGTH: int = 16


# ---------------------------------------------------------------------------
# 9.2 Ra 消息 (0x0139, 16 字节)
# ---------------------------------------------------------------------------

@dataclass
class RaMessage(_BytesMessage):
    """Ra 随机数消息 (标准 9.2)。"""
    ra: bytes  # 16 字节

    DATA_TYPE_INDEX: int = 0x0139
    BYTE_LENGTH: int = 16


# ---------------------------------------------------------------------------
# 9.2 Rb 消息 (0x013A, 16 字节)
# ---------------------------------------------------------------------------

@dataclass
class RbMessage(_BytesMessage):
    """Rb 随机数消息 (标准 9.2)。"""
    rb: bytes  # 16 字节

    DATA_TYPE_INDEX: int = 0x013A
    BYTE_LENGTH: int = 16


# ---------------------------------------------------------------------------
# 9.2 G 节点确认码 + 随机数 (0x013B, 32 字节)
# ---------------------------------------------------------------------------

@dataclass
class GNodeConfirmCodeWithRandom(_TwoBytesMessage):
    """G 节点确认码与随机数 (标准 9.2)。"""
    confirm_code: bytes  # 16 字节
    ra: bytes            # 16 字节

    DATA_TYPE_INDEX: int = 0x013B
    BYTE_LENGTH: int = 32


# ---------------------------------------------------------------------------
# 9.2 T 节点确认码 + 随机数 (0x013C, 32 字节)
# ---------------------------------------------------------------------------

@dataclass
class TNodeConfirmCodeWithRandom(_TwoBytesMessage):
    """T 节点确认码与随机数 (标准 9.2)。"""
    confirm_code: bytes  # 16 字节
    rb: bytes            # 16 字节

    DATA_TYPE_INDEX: int = 0x013C
    BYTE_LENGTH: int = 32


# ---------------------------------------------------------------------------
# 9.2 G 节点确认码 (0x013F, 16 字节)
# ---------------------------------------------------------------------------

@dataclass
class GNodeConfirmCode(_BytesMessage):
    """G 节点确认码 (标准 9.2)。"""
    confirm_code: bytes  # 16 字节

    DATA_TYPE_INDEX: int = 0x013F
    BYTE_LENGTH: int = 16


# ---------------------------------------------------------------------------
# 9.2 G 节点 DH 密钥校验 (0x0141, 16 字节)
# ---------------------------------------------------------------------------

@dataclass
class GNodeDHKeyVerify(_BytesMessage):
    """G 节点 DH 密钥校验码 (标准 9.2)。"""
    verify_code: bytes  # 16 字节

    DATA_TYPE_INDEX: int = 0x0141
    BYTE_LENGTH: int = 16


# ---------------------------------------------------------------------------
# 9.2 T 节点 DH 密钥校验 (0x0142, 16 字节)
# ---------------------------------------------------------------------------

@dataclass
class TNodeDHKeyVerify(_BytesMessage):
    """T 节点 DH 密钥校验码 (标准 9.2)。"""
    verify_code: bytes  # 16 字节

    DATA_TYPE_INDEX: int = 0x0142
    BYTE_LENGTH: int = 16


# ---------------------------------------------------------------------------
# 9.2 配对失败 (0x0147, 1 字节)
# ---------------------------------------------------------------------------

@dataclass
class PairingFailure:
    """配对失败 (标准 9.2, 表68)。

    reason 取值 0x01-0x0C:
      0x01 通行码输入失败    0x02 OOB 不可用
      0x03 鉴权请求不匹配    0x04 确认值不匹配
      0x05 不支持的配对方式   0x06 加密密钥长度不足
      0x07 不支持的命令       0x08 未确认的配对
      0x09 重复的配对尝试     0x0A 无效参数
      0x0B DHKey 校验失败     0x0C 数字比较失败
    """
    reason: int  # 0x01-0x0C

    DATA_TYPE_INDEX: int = 0x0147
    BYTE_LENGTH: int = 1

    def pack(self) -> bytes:
        return struct.pack("B", self.reason & 0xFF)

    @classmethod
    def unpack(cls, data: bytes) -> PairingFailure:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(reason=struct.unpack("B", data[:1])[0])


# ---------------------------------------------------------------------------
# 9.2 Rg 消息 (0x0148, 64 字节)
# ---------------------------------------------------------------------------

@dataclass
class RgMessage(_TwoBytesMessage):
    """Rg 随机数消息 (标准 9.2)。"""
    rg_x: bytes  # 32 字节
    rg_y: bytes  # 32 字节

    DATA_TYPE_INDEX: int = 0x0148
    BYTE_LENGTH: int = 64


# ---------------------------------------------------------------------------
# 9.2 Rt 消息 (0x0149, 64 字节)
# ---------------------------------------------------------------------------

@dataclass
class RtMessage(_TwoBytesMessage):
    """Rt 随机数消息 (标准 9.2)。"""
    rt_x: bytes  # 32 字节
    rt_y: bytes  # 32 字节

    DATA_TYPE_INDEX: int = 0x0149
    BYTE_LENGTH: int = 64


# ---------------------------------------------------------------------------
# 9.3.3 安全信息分发 — G 节点身份解析密钥 (0x0143, 18 字节)
# ---------------------------------------------------------------------------

class ResolutionAlgorithm:
    """解析算法枚举 (标准 9.3.3)。"""
    HMAC_SM3 = 0x01
    AES_CMAC_128 = 0x02


class AddrType:
    """媒体接入层标识类型 (标准 9.3.3)。"""
    UNION_ALLOCATED = 0x00
    THIRD_PARTY_LOCAL = 0x02
    UNION_RESERVED = 0x05
    PRIVATE = 0x06


@dataclass
class GNodeIRK:
    """G 节点身份解析密钥信息 (标准 9.3.3)。"""
    resolution_algorithm: int  # 1 字节
    irk: bytes                 # 16 字节 (128 bit)
    irk_id: int                # 1 字节

    DATA_TYPE_INDEX: int = 0x0143
    BYTE_LENGTH: int = 18

    def pack(self) -> bytes:
        if len(self.irk) != 16:
            raise ValueError("IRK 长度必须为 16 字节")
        return struct.pack("B", self.resolution_algorithm) + self.irk + struct.pack("B", self.irk_id)

    @classmethod
    def unpack(cls, data: bytes) -> GNodeIRK:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(
            resolution_algorithm=data[0],
            irk=bytes(data[1:17]),
            irk_id=data[17],
        )


# ---------------------------------------------------------------------------
# 9.3.3 安全信息分发 — T 节点身份解析密钥 (0x0144, 18 字节)
# ---------------------------------------------------------------------------

@dataclass
class TNodeIRK:
    """T 节点身份解析密钥信息 (标准 9.3.3)。"""
    resolution_algorithm: int
    irk: bytes  # 16 字节
    irk_id: int

    DATA_TYPE_INDEX: int = 0x0144
    BYTE_LENGTH: int = 18

    def pack(self) -> bytes:
        if len(self.irk) != 16:
            raise ValueError("IRK 长度必须为 16 字节")
        return struct.pack("B", self.resolution_algorithm) + self.irk + struct.pack("B", self.irk_id)

    @classmethod
    def unpack(cls, data: bytes) -> TNodeIRK:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(
            resolution_algorithm=data[0],
            irk=bytes(data[1:17]),
            irk_id=data[17],
        )


# ---------------------------------------------------------------------------
# 9.3.3 安全信息分发 — G 节点媒体接入层标识 (0x0145, 7 字节)
# ---------------------------------------------------------------------------

@dataclass
class GNodeAddress:
    """G 节点媒体接入层标识信息 (标准 9.3.3)。"""
    addr_type: int   # 1 字节
    addr: bytes      # 6 字节 (48 bit)

    DATA_TYPE_INDEX: int = 0x0145
    BYTE_LENGTH: int = 7

    def pack(self) -> bytes:
        if len(self.addr) != 6:
            raise ValueError("地址长度必须为 6 字节")
        return struct.pack("B", self.addr_type) + self.addr

    @classmethod
    def unpack(cls, data: bytes) -> GNodeAddress:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(addr_type=data[0], addr=bytes(data[1:7]))


# ---------------------------------------------------------------------------
# 9.3.3 安全信息分发 — T 节点媒体接入层标识 (0x0146, 7 字节)
# ---------------------------------------------------------------------------

@dataclass
class TNodeAddress:
    """T 节点媒体接入层标识信息 (标准 9.3.3)。"""
    addr_type: int
    addr: bytes  # 6 字节

    DATA_TYPE_INDEX: int = 0x0146
    BYTE_LENGTH: int = 7

    def pack(self) -> bytes:
        if len(self.addr) != 6:
            raise ValueError("地址长度必须为 6 字节")
        return struct.pack("B", self.addr_type) + self.addr

    @classmethod
    def unpack(cls, data: bytes) -> TNodeAddress:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(addr_type=data[0], addr=bytes(data[1:7]))


# ---------------------------------------------------------------------------
# 9.3.2 组播安全算法指示 (0x0148, 19 字节)
# 注: 数据类型索引与 RgMessage 相同, 协议通过状态机区分
# ---------------------------------------------------------------------------

@dataclass
class MulticastAlgorithmConfig:
    """组播安全算法指示消息 (标准 9.3.2)。"""
    rand: bytes               # 16 字节 (128 bit)
    kdf_type: int             # 1 字节
    encryption_algo: int      # 1 字节
    integrity_algo: int       # 1 字节

    DATA_TYPE_INDEX: int = 0x0148
    BYTE_LENGTH: int = 19

    def pack(self) -> bytes:
        if len(self.rand) != 16:
            raise ValueError("RAND 长度必须为 16 字节")
        return self.rand + struct.pack("BBB", self.kdf_type,
                                       self.encryption_algo,
                                       self.integrity_algo)

    @classmethod
    def unpack(cls, data: bytes) -> MulticastAlgorithmConfig:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(
            rand=bytes(data[:16]),
            kdf_type=data[16],
            encryption_algo=data[17],
            integrity_algo=data[18],
        )


# ---------------------------------------------------------------------------
# 9.3.2 组播密钥配置 (0x0149, 16 字节)
# 注: 数据类型索引与 RtMessage 相同, 协议通过状态机区分
# ---------------------------------------------------------------------------

@dataclass
class MulticastKeyConfig:
    """组播密钥配置消息 (标准 9.3.2)。"""
    c: bytes  # 16 字节 (C = Kg XOR GK)

    DATA_TYPE_INDEX: int = 0x0149
    BYTE_LENGTH: int = 16

    def pack(self) -> bytes:
        if len(self.c) != 16:
            raise ValueError("C 长度必须为 16 字节")
        return self.c

    @classmethod
    def unpack(cls, data: bytes) -> MulticastKeyConfig:
        if len(data) < cls.BYTE_LENGTH:
            raise ValueError(f"数据不足: 需要 {cls.BYTE_LENGTH} 字节")
        return cls(c=bytes(data[:16]))
