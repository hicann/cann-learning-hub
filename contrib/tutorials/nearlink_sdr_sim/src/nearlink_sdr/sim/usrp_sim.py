"""USRP 环回仿真引擎。

使用 MockUSRP loopback 模式, 将完整的 SLE 发射/接收流水线
通过虚拟 USRP 设备连接, 验证端到端功能正确性。

支持的仿真场景:
- 单帧 IQ 环回 (AWGN 信道)
- PHY 全链路环回 (CRC → Polar → 调制 → 信道 → 解调 → 解码 → CRC)
- MAC 数据帧环回 (AsyncDataFrame 打包 → PHY → 环回 → PHY → 解包)
- MAC 信令环回 (信令编码 → PHY → 环回 → PHY → 信令解码)
- 跳频发射序列
- SleNode 双节点端到端
"""

from __future__ import annotations

__all__ = [
    "LoopbackResult",
    "USRPLoopbackSim",
    "USRPSimResult",
]


import logging
from dataclasses import dataclass, field

import numpy as np

from nearlink_sdr.phy.channel import ChannelConfig, ChannelModel
from nearlink_sdr.phy.mac_interface import MacRxResult, iq_to_mac, mac_to_iq
from nearlink_sdr.phy.rx_pipeline import rx_chain
from nearlink_sdr.phy.tx_pipeline import TxConfig, tx_chain
from nearlink_sdr.phy.usrp import (
    LoopbackBuffer,
    SLETransceiver,
    USRPConfig,
    USRPDevice,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 仿真结果
# ---------------------------------------------------------------------------

@dataclass
class LoopbackResult:
    """单帧环回仿真结果。"""
    tx_samples: int
    rx_samples: int
    crc_ok: bool
    head_crc_ok: bool
    ber: float = 0.0
    snr_db: float = 0.0


@dataclass
class USRPSimResult:
    """多帧仿真汇总结果。"""
    total_frames: int = 0
    success_frames: int = 0
    failed_frames: int = 0
    avg_ber: float = 0.0
    frame_results: list[LoopbackResult] = field(default_factory=list)

    @property
    def fer(self) -> float:
        """帧错误率。"""
        if self.total_frames == 0:
            return 0.0
        return self.failed_frames / self.total_frames


# ---------------------------------------------------------------------------
# USRP 环回仿真器
# ---------------------------------------------------------------------------

class USRPLoopbackSim:
    """USRP 环回仿真器。

    在 MockUSRP loopback 模式下, TX 端发送的 IQ 信号经可选信道模型
    后直接送入 RX 端, 实现完整的硬件在环式仿真。
    """

    def __init__(
        self,
        snr_db: float = 30.0,
        channel_type: str = "awgn",
        sample_rate_hz: float = 1e6,
        seed: int = 42,
    ):
        ch_cfg = ChannelConfig(
            snr_db=snr_db,
            channel_type=channel_type,
            seed=seed,
        )
        self._channel = ChannelModel(config=ch_cfg)
        self._loopback = LoopbackBuffer(
            channel_model=self._channel,
            seed=seed,
        )
        self._usrp_cfg = USRPConfig(sample_rate_hz=sample_rate_hz)
        self._device = USRPDevice(
            config=self._usrp_cfg,
            use_mock=True,
            loopback=self._loopback,
        )
        self._xcvr = SLETransceiver(self._device)
        self._xcvr.open()

    def close(self):
        self._xcvr.close()
        self._loopback.clear()

    # ── IQ 级环回 ──

    def loopback_iq(self, iq_samples: np.ndarray) -> np.ndarray:
        """IQ 样本经 MockUSRP TX → loopback → RX 环回。"""
        self._loopback.clear()
        n_sent = self._xcvr.transmit_iq(iq_samples)
        rx_samples = self._xcvr.receive_iq(n_sent)
        return rx_samples

    # ── PHY 全链路环回 ──

    def loopback_phy(
        self,
        data_bits: np.ndarray,
        cfg: TxConfig,
        ctrl_bits: np.ndarray | None = None,
    ) -> LoopbackResult:
        """PHY 全链路环回: 编码 → 调制 → USRP TX → 信道 → USRP RX → 解调 → 解码。

        :param data_bits: 信息比特。
        :param cfg: 发射配置。
        :param ctrl_bits: 控制信息比特, None 时使用全零。

        :returns: LoopbackResult 包含 CRC 校验结果和 BER。
        """
        if ctrl_bits is None:
            ctrl_bits = np.zeros(cfg.ctrl_bits_len, dtype=np.int8)

        iq = tx_chain(ctrl_bits, data_bits, cfg)
        self._loopback.clear()

        n_sent = self._xcvr.transmit_iq(iq)
        rx_iq = self._xcvr.receive_iq(n_sent)

        n_data_bytes = (len(data_bits) + 7) // 8
        result = rx_chain(rx_iq, cfg, n_data_bytes)

        n_cmp = min(len(data_bits), len(result.data_bits))
        errors = int(np.sum(data_bits[:n_cmp] != result.data_bits[:n_cmp]))
        ber = errors / max(n_cmp, 1)

        return LoopbackResult(
            tx_samples=n_sent,
            rx_samples=len(rx_iq),
            crc_ok=result.crc_ok,
            head_crc_ok=result.head_crc_ok,
            ber=ber,
            snr_db=self._channel.cfg.snr_db,
        )

    # ── MAC 数据帧环回 ──

    def loopback_mac_data(
        self,
        mac_payload: bytes,
        cfg: TxConfig | None = None,
    ) -> tuple[MacRxResult, LoopbackResult]:
        """MAC 数据帧环回: bytes → mac_to_iq → USRP → iq_to_mac。

        :param mac_payload: MAC 层载荷字节。
        :param cfg: 发射配置, None 时使用 FT2 MCS7 默认值。

        :returns: (MacRxResult, LoopbackResult)
        """
        if cfg is None:
            cfg = TxConfig(frame_type=2, mcs_index=7)

        iq = mac_to_iq(mac_payload, cfg)
        self._loopback.clear()

        n_sent = self._xcvr.transmit_iq(iq)
        rx_iq = self._xcvr.receive_iq(n_sent)

        n_mac_bytes = len(mac_payload)
        mac_rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

        ber = 0.0
        if mac_rx.crc_ok:
            tx_bits = np.frombuffer(mac_payload, dtype=np.uint8)
            rx_bits = np.frombuffer(mac_rx.mac_payload, dtype=np.uint8)
            n_cmp = min(len(tx_bits), len(rx_bits))
            byte_errors = int(np.sum(tx_bits[:n_cmp] != rx_bits[:n_cmp]))
            ber = byte_errors / max(n_cmp, 1)

        lr = LoopbackResult(
            tx_samples=n_sent,
            rx_samples=len(rx_iq),
            crc_ok=mac_rx.crc_ok,
            head_crc_ok=mac_rx.head_crc_ok,
            ber=ber,
            snr_db=self._channel.cfg.snr_db,
        )
        return mac_rx, lr

    # ── 跳频环回 ──

    def loopback_hopping(
        self,
        iq_samples: np.ndarray,
        hop_sequence: list[int],
    ) -> list[tuple[int, np.ndarray]]:
        """跳频发射 + 接收环回。

        在每个跳频信道上发射 iq_samples, 然后在同一信道上接收。

        :returns: [(channel_num, rx_samples), ...]
        """
        results = []
        for ch in hop_sequence:
            self._loopback.clear()
            self._xcvr.hop_and_transmit(ch, iq_samples)
            rx = self._xcvr.hop_and_receive(ch, len(iq_samples))
            results.append((ch, rx))
        return results

    # ── 多帧批量仿真 ──

    def run_batch(
        self,
        payloads: list[bytes],
        cfg: TxConfig | None = None,
    ) -> USRPSimResult:
        """批量 MAC 帧环回仿真。

        :param payloads: MAC 载荷列表。
        :param cfg: 发射配置。

        :returns: USRPSimResult 汇总结果。
        """
        summary = USRPSimResult()
        for payload in payloads:
            _, lr = self.loopback_mac_data(payload, cfg)
            summary.total_frames += 1
            if lr.crc_ok:
                summary.success_frames += 1
            else:
                summary.failed_frames += 1
            summary.frame_results.append(lr)

        if summary.frame_results:
            summary.avg_ber = float(np.mean(
                [r.ber for r in summary.frame_results]
            ))
        return summary

    @property
    def device(self) -> USRPDevice:
        return self._device

    @property
    def transceiver(self) -> SLETransceiver:
        return self._xcvr
