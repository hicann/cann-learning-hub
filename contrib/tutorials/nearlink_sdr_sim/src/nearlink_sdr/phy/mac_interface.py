"""MAC-PHY 集成适配层。

将 MAC 层帧对象与 PHY 层发射/接收流水线打通，
提供字节级到比特级的转换和完整的发收链路。

QoS 集成: QosLink 类封装了带 ARQ/HARQ/流控的端到端数据链路。
"""

from __future__ import annotations

import struct

__all__ = [
    "MacRxResult",
    "QosLink",
    "bits_to_bytes",
    "bytes_to_bits",
    "iq_to_mac",
    "iq_to_signaling",
    "mac_to_iq",
    "roundtrip_data",
    "roundtrip_signaling",
    "signaling_to_iq",
]


from dataclasses import dataclass

import numpy as np

from nearlink_sdr.mac.frame import (
    AsyncDataFrame,
    ControlFrame,
)
from nearlink_sdr.mac.qos import (
    ArqState,
    FlowController,
    HarqController,
    LinkQualityTracker,
    Priority,
    QosManager,
    TxDecision,
    TxQueue,
)
from nearlink_sdr.mac.signaling import decode_signaling, encode_signaling
from nearlink_sdr.phy.control_info import ControlInfoA2
from nearlink_sdr.phy.rx_pipeline import rx_chain
from nearlink_sdr.phy.tx_pipeline import TxConfig, tx_chain

# -----------------------------------------------------------------------
# 字节 / 比特转换
# -----------------------------------------------------------------------

def bytes_to_bits(data: bytes) -> np.ndarray:
    """字节序列 → 比特数组 (MSB first)。"""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """比特数组 → 字节序列 (长度向上对齐到 8 的倍数)。"""
    n = len(bits)
    padded = bits
    if n % 8 != 0:
        padded = np.concatenate([bits, np.zeros(8 - n % 8, dtype=np.uint8)])
    return np.packbits(padded.astype(np.uint8)).tobytes()


# -----------------------------------------------------------------------
# 发射端适配
# -----------------------------------------------------------------------

def mac_to_iq(
    mac_payload: bytes,
    cfg: TxConfig,
    ctrl_info: ControlInfoA2 | None = None,
) -> np.ndarray:
    """MAC 层载荷 → 基带 IQ 信号。

    :param mac_payload: MAC 帧 pack() 产生的字节流。
    :param cfg: 发射参数配置。
    :param ctrl_info: 物理层控制信息。若为 None 则使用默认 A2 配置。
    :returns: IQ 采样数组 (complex128)。
    """
    data_bits = bytes_to_bits(mac_payload)

    if ctrl_info is None:
        ctrl_info = ControlInfoA2(
            packet_type=0,
            empty_packet=0,
            tx_sn=0,
            rx_sn=0,
            flow_ctrl=0,
            sys_mgmt_rx=0,
            reserved=0,
            data_length=len(mac_payload),
        )

    ctrl_bits = ctrl_info.pack(sync_seed=cfg.pid)

    return tx_chain(ctrl_bits, data_bits, cfg)


def signaling_to_iq(msg: object, cfg: TxConfig) -> np.ndarray:
    """信令对象 → 基带 IQ 信号。

    将一条 MAC 信令消息编码为 ControlFrame -> bytes -> bits -> tx_chain。
    """
    frame = encode_signaling(msg)
    mac_bytes = frame.pack()
    return mac_to_iq(mac_bytes, cfg)


# -----------------------------------------------------------------------
# 接收端适配
# -----------------------------------------------------------------------

@dataclass
class MacRxResult:
    """MAC 层接收结果。"""
    mac_payload: bytes
    ctrl_bits: np.ndarray
    crc_ok: bool
    head_crc_ok: bool


def iq_to_mac(
    iq_signal: np.ndarray,
    cfg: TxConfig,
    n_mac_bytes: int,
) -> MacRxResult:
    """基带 IQ 信号 → MAC 层载荷字节。

    :param iq_signal: IQ 采样数组。
    :param cfg: 与发射端相同的配置。
    :param n_mac_bytes: MAC 层载荷字节数（需要预先知道）。
    :returns: MacRxResult。
    """
    result = rx_chain(iq_signal, cfg, n_mac_bytes)
    mac_payload = bits_to_bytes(result.data_bits[:n_mac_bytes * 8])
    return MacRxResult(
        mac_payload=mac_payload,
        ctrl_bits=result.ctrl_bits,
        crc_ok=result.crc_ok,
        head_crc_ok=result.head_crc_ok,
    )


def iq_to_signaling(
    iq_signal: np.ndarray,
    cfg: TxConfig,
    n_mac_bytes: int,
) -> tuple[object, bool]:
    """基带 IQ 信号 → 信令对象。

    :returns: (信令对象, CRC是否通过)。
    """
    rx = iq_to_mac(iq_signal, cfg, n_mac_bytes)
    if not rx.crc_ok:
        return None, False
    frame, _ = ControlFrame.unpack(rx.mac_payload)
    msg = decode_signaling(frame)
    return msg, True


# -----------------------------------------------------------------------
# 全链路 roundtrip (用于测试/仿真)
# -----------------------------------------------------------------------

def roundtrip_signaling(
    msg: object,
    cfg: TxConfig | None = None,
) -> tuple[object | None, bool]:
    """信令编码 → IQ → 解码全链路测试。

    无信道（直连），验证 MAC → PHY → MAC 数据完整性。

    :returns: (恢复的信令对象, CRC是否通过)。
    """
    if cfg is None:
        cfg = TxConfig(frame_type=2, mcs_index=7)

    frame = encode_signaling(msg)
    mac_bytes = frame.pack()
    n_mac_bytes = len(mac_bytes)

    iq = mac_to_iq(mac_bytes, cfg)
    return iq_to_signaling(iq, cfg, n_mac_bytes)


def roundtrip_data(
    data: bytes,
    cfg: TxConfig | None = None,
    segment_type: int = 0,
) -> tuple[bytes, bool]:
    """异步数据帧编码 → IQ → 解码全链路测试。

    :returns: (恢复的数据, CRC是否通过)。
    """
    if cfg is None:
        cfg = TxConfig(frame_type=2, mcs_index=7)

    frame = AsyncDataFrame(segment_type=segment_type, data=data)
    mac_bytes = frame.pack()
    n_mac_bytes = len(mac_bytes)

    iq = mac_to_iq(mac_bytes, cfg)
    rx = iq_to_mac(iq, cfg, n_mac_bytes)

    if not rx.crc_ok:
        return b"", False

    recovered = AsyncDataFrame.unpack(rx.mac_payload)
    return recovered.data, True


# -----------------------------------------------------------------------
# QoS 集成链路
# -----------------------------------------------------------------------


class QosLink:
    """带 QoS 管理的端到端数据链路。

    封装 ARQ 序列号管理、HARQ 反馈、流控和链路质量跟踪。
    调用方通过 submit / transmit / receive 方法实现自动重传驱动。

    :param cfg: 发射参数配置。
    :param frame_type: 帧类型 (1-4), 决定 ARQ SN 位宽。
    :param max_pdu: 单 PDU 最大字节数。
    """

    def __init__(
        self,
        cfg: TxConfig | None = None,
        frame_type: int = 2,
        max_pdu: int = 256,
    ) -> None:
        if cfg is None:
            cfg = TxConfig(frame_type=frame_type, mcs_index=7)
        self.cfg = cfg
        self.max_pdu = max_pdu
        self.qos = QosManager(
            arq=ArqState(frame_type=frame_type),
            harq=HarqController(),
            flow=FlowController(),
            quality=LinkQualityTracker(),
            tx_queue=TxQueue(),
        )
        self._tx_count = 0
        self._rx_count = 0
        self._retx_count = 0

    def submit(
        self, data: bytes, priority: Priority = Priority.NORMAL
    ) -> bool:
        """提交数据到发送队列。"""
        return self.qos.submit_data(data, priority)

    def transmit(self) -> tuple[np.ndarray | None, TxDecision]:
        """从队列取数据并生成 IQ 信号。

        :returns: (IQ 信号或 None, 发送决策)。
        """
        decision, item = self.qos.prepare_tx()

        if decision == TxDecision.EMPTY:
            return None, decision

        if decision == TxDecision.NEW_DATA and item is not None:
            data = item.data
        elif decision == TxDecision.RETRANSMIT:
            data = self.qos.arq._last_tx_data
        else:
            data = self.qos.arq._last_tx_data

        if not data:
            return None, TxDecision.EMPTY

        frame = AsyncDataFrame(segment_type=0, data=data)
        mac_bytes = frame.pack()

        fields = self.qos.get_ctrl_fields()
        ctrl = ControlInfoA2(
            packet_type=fields.get("packet_type", 0),
            empty_packet=fields["empty_packet"],
            tx_sn=fields["tx_sn"],
            rx_sn=fields["rx_sn"],
            flow_ctrl=fields["flow_ctrl"],
            sys_mgmt_rx=0,
            reserved=0,
            data_length=len(mac_bytes),
        )

        iq = mac_to_iq(mac_bytes, self.cfg, ctrl_info=ctrl)
        self._tx_count += 1
        if decision == TxDecision.RETRANSMIT:
            self._retx_count += 1

        return iq, decision

    def receive(self, iq: np.ndarray, n_mac_bytes: int) -> tuple[bytes, bool]:
        """接收 IQ 信号并处理 ARQ 反馈。

        :returns: (恢复的数据, CRC 是否通过)。
        """
        rx = iq_to_mac(iq, self.cfg, n_mac_bytes)
        self._rx_count += 1

        self.qos.on_tx_feedback(rx.crc_ok)

        if not rx.crc_ok:
            return b"", False

        try:
            recovered = AsyncDataFrame.unpack(rx.mac_payload)
            return recovered.data, True
        except (ValueError, struct.error, IndexError):
            return b"", False

    def process_feedback(self, crc_ok: bool) -> TxDecision:
        """外部反馈处理 (对端发来 CRC 结果)。"""
        return self.qos.on_tx_feedback(crc_ok)

    @property
    def pending_retransmit(self) -> bool:
        """当前是否有待重传数据。"""
        return self.qos.arq.pending_ack

    @property
    def recommended_mcs(self) -> int:
        """基于链路质量建议的 MCS。"""
        return self.qos.quality.apply_suggestion()

    @property
    def stats(self) -> dict[str, int | float]:
        """链路统计信息。"""
        return {
            "tx_count": self._tx_count,
            "rx_count": self._rx_count,
            "retx_count": self._retx_count,
            "queue_size": self.qos.tx_queue.size,
            "fer": self.qos.quality.fer,
            "current_mcs": self.qos.quality.current_mcs,
        }
