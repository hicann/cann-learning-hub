"""安全流程集成管理 -- TXS-10002-2025 标准 9.2-9.4。

将配对信令 (security.py)、密码学原语 (crypto.py) 和链路管理器
整合为端到端的安全流程:
- 配对状态机 (PairingManager): 驱动配对信令交互和密钥协商
- 帧加密上下文 (FrameCryptoContext): 管理 payload_count 和 AES-CCM 加解密
- ECDH 密钥交换 (P-256): 生成密钥对和共享密钥
"""

from __future__ import annotations

__all__ = [
    "ECDHKeyPair",
    "FrameCryptoContext",
    "PairingFailureReason",
    "PairingManager",
    "PairingState",
    "run_pairing_procedure",
]


import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum, auto

from nearlink_sdr.mac.crypto import (
    AuthMethod,
    KdfType,
    aes_ccm_decrypt,
    aes_ccm_encrypt,
    build_ccm_nonce_async,
    compute_iv,
    derive_link_key,
    derive_session_key,
    generate_confirm_code,
)
from nearlink_sdr.mac.security import (
    GNodeConfirmCode,
    PairingConfirm,
    PairingFailure,
    PairingInitialInfo,
    PairingInitiate,
    PairingRequest,
    PairingResponse,
    RaMessage,
    RbMessage,
    TNodeConfirmCode,
)

log = logging.getLogger(__name__)


# ── 配对状态 ──


class PairingState(IntEnum):
    """配对流程状态。"""
    IDLE = 0
    INITIATED = auto()
    REQUEST_SENT = auto()
    RESPONSE_SENT = auto()
    CONFIRM_SENT = auto()
    PUBLIC_KEY_EXCHANGED = auto()
    CONFIRM_CODE_SENT = auto()
    COMPLETED = auto()
    FAILED = auto()


class PairingFailureReason(IntEnum):
    """配对失败原因 (标准 9.2)。"""
    PASSKEY_ENTRY_FAILED = 0x01
    OOB_NOT_AVAILABLE = 0x02
    AUTHENTICATION_REQUIREMENTS = 0x03
    CONFIRM_VALUE_FAILED = 0x04
    PAIRING_NOT_SUPPORTED = 0x05
    ENCRYPTION_KEY_SIZE = 0x06
    COMMAND_NOT_SUPPORTED = 0x07
    UNSPECIFIED_REASON = 0x08
    REPEATED_ATTEMPTS = 0x09
    INVALID_PARAMETERS = 0x0A
    DHKEY_CHECK_FAILED = 0x0B
    NUMERIC_COMPARISON_FAILED = 0x0C


# ── ECDH P-256 密钥交换 ──


@dataclass
class ECDHKeyPair:
    """P-256 椭圆曲线密钥对。"""
    private_key_bytes: bytes = b""
    public_key_x: bytes = b""
    public_key_y: bytes = b""

    @classmethod
    def generate(cls) -> ECDHKeyPair:
        """生成 P-256 密钥对。"""
        from cryptography.hazmat.primitives.asymmetric.ec import (
            SECP256R1,
            generate_private_key,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )

        private_key = generate_private_key(SECP256R1())
        private_bytes = private_key.private_bytes(
            Encoding.DER, PrivateFormat.PKCS8, NoEncryption(),
        )
        public_bytes = private_key.public_key().public_bytes(
            Encoding.X962, PublicFormat.UncompressedPoint,
        )
        # X962 uncompressed: 0x04 || X(32B) || Y(32B)
        pub_x = public_bytes[1:33]
        pub_y = public_bytes[33:65]
        return cls(
            private_key_bytes=private_bytes,
            public_key_x=pub_x,
            public_key_y=pub_y,
        )

    def compute_shared_secret(self, peer_pub_x: bytes, peer_pub_y: bytes) -> bytes:
        """计算 ECDH 共享密钥。

        :returns: 共享密钥 (32 字节)。
        """
        from cryptography.hazmat.primitives.asymmetric.ec import (
            ECDH,
            SECP256R1,
            EllipticCurvePublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            load_der_private_key,
        )

        private_key = load_der_private_key(self.private_key_bytes, password=None)
        # 构建对端公钥
        peer_pub_bytes = b"\x04" + peer_pub_x + peer_pub_y
        peer_public_key = EllipticCurvePublicKey.from_encoded_point(
            SECP256R1(), peer_pub_bytes,
        )
        shared_key = private_key.exchange(ECDH(), peer_public_key)
        return shared_key


# ── 配对管理器 ──


@dataclass
class PairingManager:
    """配对流程状态机 (标准 9.2)。

    驱动 G 节点与 T 节点之间的配对信令交互, 完成:
    1. 配对发起/请求/响应
    2. ECDH 公钥交换
    3. 确认码互验
    4. 链路密钥派生
    5. 会话密钥派生

    :ivar is_g_node: 本端是否为 G 节点。
    :ivar kdf_type: 密钥派生函数类型。
    :ivar auth_method: 鉴权方式。
    :ivar max_key_length: 最大密钥长度 (字节)。
    """
    is_g_node: bool = True
    kdf_type: KdfType = KdfType.AES_CMAC
    auth_method: AuthMethod = AuthMethod.NUMERIC_COMPARISON
    max_key_length: int = 16
    local_address: bytes = b"\x00" * 6
    peer_address: bytes = b"\x00" * 6

    # 状态
    state: PairingState = PairingState.IDLE
    failure_reason: PairingFailureReason | None = None

    # 密钥材料
    local_keypair: ECDHKeyPair = field(default_factory=ECDHKeyPair)
    peer_pub_x: bytes = b""
    peer_pub_y: bytes = b""
    local_random: bytes = b""
    peer_random: bytes = b""
    dh_key: bytes = b""
    link_key: bytes = b""
    session_key: bytes = b""
    integrity_key: bytes | None = None

    # 信令历史
    _outgoing: list[object] = field(default_factory=list)

    def start_pairing(self) -> list[object]:
        """发起配对, 返回待发送的信令列表。

        G 节点发送: PairingInitiate → PairingRequest
        T 节点发送: (等待 G 节点发起)
        """
        self.state = PairingState.IDLE
        self._outgoing.clear()

        # 生成本端密钥对
        self.local_keypair = ECDHKeyPair.generate()
        # 生成本端随机数
        self.local_random = os.urandom(16)

        if self.is_g_node:
            initiate = PairingInitiate(auth_request=0x01)
            request = PairingRequest(
                io_capability=0x03,
                oob_data_flag=0,
                auth_request=0x01,
                max_key_length=self.max_key_length,
                security_dist_info=0x01,
                crypto_capability=b"\x01\x00\x00\x00",
                psk_indication=0,
            )
            self._outgoing.extend([initiate, request])
            self.state = PairingState.REQUEST_SENT
        else:
            # T 节点等待接收
            self.state = PairingState.INITIATED

        return list(self._outgoing)

    def process_message(self, msg: object) -> list[object]:
        """处理收到的配对信令, 返回待发送的响应信令。"""
        responses: list[object] = []

        if isinstance(msg, PairingInitiate):
            responses = self._handle_initiate(msg)
        elif isinstance(msg, PairingRequest):
            responses = self._handle_request(msg)
        elif isinstance(msg, PairingResponse):
            responses = self._handle_response(msg)
        elif isinstance(msg, PairingConfirm):
            responses = self._handle_confirm(msg)
        elif isinstance(msg, PairingInitialInfo):
            responses = self._handle_initial_info(msg)
        elif isinstance(msg, RaMessage):
            responses = self._handle_ra(msg)
        elif isinstance(msg, RbMessage):
            responses = self._handle_rb(msg)
        elif isinstance(msg, TNodeConfirmCode):
            responses = self._handle_t_confirm(msg)
        elif isinstance(msg, GNodeConfirmCode):
            responses = self._handle_g_confirm(msg)
        elif isinstance(msg, PairingFailure):
            self.state = PairingState.FAILED
            self.failure_reason = PairingFailureReason(msg.reason)
            log.warning("配对失败: %s", self.failure_reason.name)

        self._outgoing.extend(responses)
        return responses

    def _handle_initiate(self, msg: PairingInitiate) -> list[object]:
        if not self.is_g_node:
            self.state = PairingState.INITIATED
        return []

    def _handle_request(self, msg: PairingRequest) -> list[object]:
        if not self.is_g_node:
            # T 节点: 响应配对请求
            response = PairingResponse(
                io_capability=0x03,
                oob_data_flag=0,
                auth_request=0x01,
                max_key_length=self.max_key_length,
                security_dist_info=0x01,
                crypto_capability=b"\x01\x00\x00\x00",
                psk_indication=0,
            )
            self.state = PairingState.RESPONSE_SENT
            return [response]
        return []

    def _handle_response(self, msg: PairingResponse) -> list[object]:
        if self.is_g_node:
            # G 节点: 发送 PairingConfirm (含公钥)
            confirm = PairingConfirm(
                key_length=self.max_key_length,
                auth_method=int(self.auth_method),
                crypto_algorithm=b"\x01\x00\x00\x00",
                g_public_key_x=self.local_keypair.public_key_x,
                g_public_key_y=self.local_keypair.public_key_y,
            )
            self.state = PairingState.CONFIRM_SENT
            return [confirm]
        return []

    def _handle_confirm(self, msg: PairingConfirm) -> list[object]:
        if not self.is_g_node:
            # T 节点: 保存 G 公钥, 发送 T 公钥
            self.peer_pub_x = msg.g_public_key_x
            self.peer_pub_y = msg.g_public_key_y
            initial_info = PairingInitialInfo(
                t_public_key_x=self.local_keypair.public_key_x,
                t_public_key_y=self.local_keypair.public_key_y,
            )
            self.state = PairingState.PUBLIC_KEY_EXCHANGED
            return [initial_info]
        return []

    def _handle_initial_info(self, msg: PairingInitialInfo) -> list[object]:
        if self.is_g_node:
            # G 节点: 保存 T 公钥, 计算 DH Key, 发送 Ra
            self.peer_pub_x = msg.t_public_key_x
            self.peer_pub_y = msg.t_public_key_y
            self._compute_dh_key()
            ra = RaMessage(ra=self.local_random)
            self.state = PairingState.PUBLIC_KEY_EXCHANGED
            return [ra]
        return []

    def _handle_ra(self, msg: RaMessage) -> list[object]:
        if not self.is_g_node:
            # T 节点: 保存 Ra, 计算 DH Key, 发送 Rb
            self.peer_random = msg.ra
            self._compute_dh_key()
            rb = RbMessage(rb=self.local_random)
            return [rb]
        return []

    def _handle_rb(self, msg: RbMessage) -> list[object]:
        if self.is_g_node:
            # G 节点: 保存 Rb, 生成并发送确认码
            self.peer_random = msg.rb
            confirm_code = self._generate_confirm_code()
            g_confirm = GNodeConfirmCode(confirm_code=confirm_code)
            self.state = PairingState.CONFIRM_CODE_SENT
            return [g_confirm]
        return []

    def _handle_t_confirm(self, msg: TNodeConfirmCode) -> list[object]:
        if self.is_g_node:
            # G 节点: 验证 T 确认码
            expected = self._generate_peer_confirm_code()
            if msg.confirm_code == expected:
                self._derive_keys()
                self.state = PairingState.COMPLETED
                log.info("配对完成 (G 节点)")
            else:
                self.state = PairingState.FAILED
                self.failure_reason = PairingFailureReason.CONFIRM_VALUE_FAILED
                return [PairingFailure(reason=0x04)]
        return []

    def _handle_g_confirm(self, msg: GNodeConfirmCode) -> list[object]:
        if not self.is_g_node:
            # T 节点: 验证 G 确认码, 发送 T 确认码
            expected = self._generate_peer_confirm_code()
            if msg.confirm_code == expected:
                confirm_code = self._generate_confirm_code()
                t_confirm = TNodeConfirmCode(confirm_code=confirm_code)
                self._derive_keys()
                self.state = PairingState.COMPLETED
                log.info("配对完成 (T 节点)")
                return [t_confirm]
            else:
                self.state = PairingState.FAILED
                self.failure_reason = PairingFailureReason.CONFIRM_VALUE_FAILED
                return [PairingFailure(reason=0x04)]
        return []

    def _compute_dh_key(self) -> None:
        """计算 ECDH 共享密钥。"""
        self.dh_key = self.local_keypair.compute_shared_secret(
            self.peer_pub_x, self.peer_pub_y,
        )

    def _generate_confirm_code(self) -> bytes:
        """生成本端确认码。"""
        if self.is_g_node:
            g_pub = self.local_keypair.public_key_x + self.local_keypair.public_key_y
            t_pub = self.peer_pub_x + self.peer_pub_y
        else:
            g_pub = self.peer_pub_x + self.peer_pub_y
            t_pub = self.local_keypair.public_key_x + self.local_keypair.public_key_y

        return generate_confirm_code(
            self.kdf_type,
            self.auth_method,
            self.local_random,
            g_pub,
            t_pub,
        )

    def _generate_peer_confirm_code(self) -> bytes:
        """生成对端确认码 (用于验证)。"""
        if self.is_g_node:
            g_pub = self.local_keypair.public_key_x + self.local_keypair.public_key_y
            t_pub = self.peer_pub_x + self.peer_pub_y
        else:
            g_pub = self.peer_pub_x + self.peer_pub_y
            t_pub = self.local_keypair.public_key_x + self.local_keypair.public_key_y

        return generate_confirm_code(
            self.kdf_type,
            self.auth_method,
            self.peer_random,
            g_pub,
            t_pub,
        )

    def _derive_keys(self) -> None:
        """从 DH Key 派生链路密钥和会话密钥。"""
        ra = self.local_random if self.is_g_node else self.peer_random
        rb = self.peer_random if self.is_g_node else self.local_random
        g_addr = self.local_address if self.is_g_node else self.peer_address
        t_addr = self.peer_address if self.is_g_node else self.local_address

        self.link_key = derive_link_key(
            self.kdf_type, self.dh_key, ra, rb, g_addr, t_addr,
        )

        # 确定性生成分散因子: 从 Ra 和 Rb 截取前 8 字节
        g_div = ra[:8]
        t_div = rb[:8]
        self.session_key, self.integrity_key = derive_session_key(
            self.kdf_type, self.link_key, g_div, t_div,
            use_authenticated=True,
        )

    @property
    def is_paired(self) -> bool:
        return self.state == PairingState.COMPLETED


# ── 帧加密上下文 ──


@dataclass
class FrameCryptoContext:
    """帧级 AES-CCM 加解密上下文。

    管理 payload_count 单调递增和 nonce 构建,
    为 MAC 帧提供加解密服务。

    :ivar session_key: 会话密钥 (16 字节)。
    :ivar iv_base: 初始化向量基底 (8 字节)。
    :ivar direction: 传输方向 (0=G→T, 1=T→G)。
    :ivar mic_len: MIC 长度 (字节, 4/8/12/16)。
    :ivar frame_type: 帧类型 (用于 IV 计算)。
    :ivar link_id: 链路标识 (用于 IV 计算)。
    """
    session_key: bytes = b"\x00" * 16
    iv_base: bytes = b"\x00" * 8
    direction: int = 0
    mic_len: int = 4
    frame_type: int = 2
    link_id: int = 0

    _tx_payload_count: int = 0
    _rx_payload_count: int = 0

    @property
    def tx_count(self) -> int:
        return self._tx_payload_count

    @property
    def rx_count(self) -> int:
        return self._rx_payload_count

    def _build_nonce(self, payload_count: int, data_length: int) -> bytes:
        """构建 CCM nonce (13 字节)。

        build_ccm_nonce_async 返回 16 字节 B0 块:
        flag(1B) | nonce(13B) | data_length(2B)
        AESCCM 需要 13 字节 nonce 部分。
        """
        iv = compute_iv(self.iv_base, self.link_id, self.frame_type)
        b0_block = build_ccm_nonce_async(
            payload_count=payload_count,
            direction=self.direction,
            iv_base=iv,
            data_length=data_length,
        )
        return b0_block[1:14]

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
        """加密一帧数据, 自动递增 payload_count。

        :param plaintext: 明文数据。
        :param aad: 关联数据 (不加密但参与完整性校验)。

        :returns: (密文, MIC)。
        """
        nonce = self._build_nonce(self._tx_payload_count, len(plaintext))
        ciphertext, mic = aes_ccm_encrypt(
            self.session_key, nonce, plaintext, aad, self.mic_len,
        )
        self._tx_payload_count += 1
        return ciphertext, mic

    def decrypt(self, ciphertext: bytes, mic: bytes, aad: bytes = b"") -> bytes:
        """解密一帧数据, 自动递增 payload_count。

        :param ciphertext: 密文数据。
        :param mic: 消息完整性码。
        :param aad: 关联数据。

        :returns: 解密后的明文。

        """
        nonce = self._build_nonce(self._rx_payload_count, len(ciphertext))
        plaintext = aes_ccm_decrypt(
            self.session_key, nonce, ciphertext, mic, aad, self.mic_len,
        )
        self._rx_payload_count += 1
        return plaintext

    def reset_counters(self) -> None:
        """重置 payload 计数器。"""
        self._tx_payload_count = 0
        self._rx_payload_count = 0


# ── 端到端配对流程 (仿真/测试用) ──


def run_pairing_procedure(
    g_address: bytes = b"\x01\x02\x03\x04\x05\x06",
    t_address: bytes = b"\x0A\x0B\x0C\x0D\x0E\x0F",
    kdf_type: KdfType = KdfType.AES_CMAC,
    auth_method: AuthMethod = AuthMethod.NUMERIC_COMPARISON,
) -> tuple[PairingManager, PairingManager]:
    """执行完整的端到端配对流程。

    模拟 G 节点和 T 节点之间的配对信令交互,
    完成密钥交换和会话密钥派生。

    :returns: (G 节点管理器, T 节点管理器)。
    """
    g_mgr = PairingManager(
        is_g_node=True,
        kdf_type=kdf_type,
        auth_method=auth_method,
        local_address=g_address,
        peer_address=t_address,
    )
    t_mgr = PairingManager(
        is_g_node=False,
        kdf_type=kdf_type,
        auth_method=auth_method,
        local_address=t_address,
        peer_address=g_address,
    )

    # G 节点发起
    g_msgs = g_mgr.start_pairing()
    # T 节点也准备密钥
    t_mgr.start_pairing()

    # G 产生的消息发给 T, T 产生的消息发给 G
    # 用 (source, msg) 元组跟踪消息来源
    pending: list[tuple[str, object]] = [("g", m) for m in g_msgs]
    max_rounds = 20

    for _ in range(max_rounds):
        if not pending:
            break

        next_pending: list[tuple[str, object]] = []
        for source, msg in pending:
            if source == "g":
                # G 发送的消息由 T 处理
                responses = t_mgr.process_message(msg)
                next_pending.extend(("t", r) for r in responses)
            else:
                # T 发送的消息由 G 处理
                responses = g_mgr.process_message(msg)
                next_pending.extend(("g", r) for r in responses)

        pending = next_pending

        if (g_mgr.state == PairingState.COMPLETED
                and t_mgr.state == PairingState.COMPLETED):
            break

    return g_mgr, t_mgr
