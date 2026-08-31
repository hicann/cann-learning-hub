"""Mock SDR 后端 -- 无硬件仿真。

从原 usrp.py 中的 MockUSRP / LoopbackBuffer 逻辑提取而来,
实现 SDRDevice 接口, 用于无硬件环境的开发与测试。
"""

from __future__ import annotations

__all__ = [
    "LoopbackBuffer",
    "MockDevice",
]

import logging

import numpy as np

from nearlink_sdr.phy.sdr_backend import SDRConfig, SDRDevice

logger = logging.getLogger(__name__)


class LoopbackBuffer:
    """TX->RX 环回缓冲区, 可选信道损伤。"""

    def __init__(self, channel_model: object | None = None, seed: int = 42):
        self._buffer: list[np.ndarray] = []
        self._channel = channel_model
        self._rng = np.random.default_rng(seed)

    def push(self, samples: np.ndarray) -> None:
        """TX 端写入样本。"""
        sig = np.asarray(samples, dtype=np.complex64).ravel()
        if self._channel is not None:
            sig = np.asarray(self._channel.apply_awgn(sig), dtype=np.complex64)
        self._buffer.append(sig)

    def pull(self, n: int) -> np.ndarray:
        """RX 端读取最多 n 个样本。"""
        if not self._buffer:
            return np.array([], dtype=np.complex64)
        head = self._buffer[0]
        if len(head) <= n:
            self._buffer.pop(0)
            return head
        out = head[:n]
        self._buffer[0] = head[n:]
        return out

    @property
    def available(self) -> int:
        return sum(len(b) for b in self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


class MockDevice(SDRDevice):
    """模拟 SDR 设备, 用于无硬件测试。

    支持环回模式: TX 发射的数据经可选信道模型后送入 RX。
    """

    def __init__(
        self,
        config: SDRConfig,
        loopback: LoopbackBuffer | None = None,
    ):
        super().__init__(config)
        self._loopback = loopback
        self._rng = np.random.default_rng(42)
        self._freq_hz = config.center_freq_hz
        self._sample_rate_hz = config.sample_rate_hz
        self._rx_gain_db = config.rx_gain_db
        self._tx_gain_db = config.tx_gain_db
        self._bw_hz = config.bandwidth_hz
        self.configure()

    def configure(self) -> None:
        cfg = self._config
        self._freq_hz = cfg.center_freq_hz
        self._sample_rate_hz = cfg.sample_rate_hz
        self._rx_gain_db = cfg.rx_gain_db
        self._tx_gain_db = cfg.tx_gain_db
        self._bw_hz = cfg.bandwidth_hz
        logger.info("MockDevice 配置完成: %s", self.status_string())

    def set_frequency(self, freq_hz: float) -> None:
        self._freq_hz = freq_hz

    def set_sample_rate(self, rate_hz: float) -> None:
        self._sample_rate_hz = rate_hz
        self._config.sample_rate_hz = rate_hz

    def set_rx_gain(self, gain_db: float) -> None:
        self._rx_gain_db = gain_db
        self._config.rx_gain_db = gain_db

    def set_tx_gain(self, gain_db: float) -> None:
        self._tx_gain_db = gain_db
        self._config.tx_gain_db = gain_db

    def set_bandwidth(self, bw_hz: float) -> None:
        self._bw_hz = bw_hz

    def transmit(self, samples: np.ndarray) -> int:
        data = np.ascontiguousarray(samples.astype(np.complex64))
        if self._loopback is not None:
            self._loopback.push(data)
        return len(data)

    def receive(self, num_samps: int) -> np.ndarray:
        if self._loopback is not None and self._loopback.available > 0:
            chunks = []
            remaining = num_samps
            while remaining > 0 and self._loopback.available > 0:
                chunk = self._loopback.pull(remaining)
                chunks.append(chunk)
                remaining -= len(chunk)
            result = np.concatenate(chunks) if chunks else np.array([], dtype=np.complex64)
            if len(result) < num_samps:
                pad = np.zeros(num_samps - len(result), dtype=np.complex64)
                result = np.concatenate([result, pad])
            return result[:num_samps]

        # 无环回时返回噪声
        noise = (
            self._rng.standard_normal(num_samps)
            + 1j * self._rng.standard_normal(num_samps)
        ).astype(np.complex64) * 0.01
        return noise

    def close(self) -> None:
        if self._loopback is not None:
            self._loopback.clear()
        logger.info("MockDevice 已关闭")

    def status_string(self) -> str:
        cfg = self._config
        return (
            f"[Mock] CH{cfg.channel_num} ({cfg.band}) "
            f"Fc={self._freq_hz / 1e6:.1f}MHz "
            f"Rate={self._sample_rate_hz / 1e6:.1f}Msps "
            f"RXG={self._rx_gain_db:.1f}dB TXG={self._tx_gain_db:.1f}dB"
        )

    @property
    def loopback(self) -> LoopbackBuffer | None:
        return self._loopback
