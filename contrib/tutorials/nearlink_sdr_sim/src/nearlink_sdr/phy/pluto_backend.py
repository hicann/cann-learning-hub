"""PlutoSDR / ANTSDR 后端 -- 基于 pyadi-iio。

支持 MicroPhase ANTSDR E310 及 Analog Devices ADALM-Pluto 等
基于 AD9361 + libiio 的 SDR 设备。
"""

from __future__ import annotations

__all__ = [
    "PLUTO_FREQ_MAX_HZ",
    "PLUTO_FREQ_MIN_HZ",
    "PLUTO_RX_GAIN_MAX",
    "PLUTO_RX_GAIN_MIN",
    "PLUTO_SAMPLE_RATE_MAX",
    "PLUTO_TX_ATTN_MAX",
    "PLUTO_TX_ATTN_MIN",
    "PlutoDevice",
    "adi_available",
]

import logging

import numpy as np

from nearlink_sdr.phy.sdr_backend import SDRConfig, SDRDevice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pyadi-iio 条件导入
# ---------------------------------------------------------------------------

try:
    import adi

    _ADI_AVAILABLE = True
except ImportError:
    adi = None  # type: ignore[assignment]
    _ADI_AVAILABLE = False


def adi_available() -> bool:
    """检查当前环境是否安装了 pyadi-iio。"""
    return _ADI_AVAILABLE


# ---------------------------------------------------------------------------
# PlutoSDR / ANTSDR 硬件常量
# ---------------------------------------------------------------------------

# 默认 Pluto: 325 MHz - 3.8 GHz; ANTSDR E310 扩展后: 70 MHz - 6 GHz
PLUTO_FREQ_MIN_HZ = 70e6
PLUTO_FREQ_MAX_HZ = 6e9
PLUTO_SAMPLE_RATE_MAX = 61.44e6  # AD9361 最大采样率
PLUTO_RX_GAIN_MIN = -3.0
PLUTO_RX_GAIN_MAX = 71.0
# Pluto TX 使用衰减值 (负数 dB), 范围 -89.75 到 0
PLUTO_TX_ATTN_MIN = -89.75
PLUTO_TX_ATTN_MAX = 0.0


class PlutoDevice(SDRDevice):
    """PlutoSDR / ANTSDR E310 设备接口。

    基于 pyadi-iio 的 adi.Pluto 类, 提供 SDRDevice 统一接口。
    """

    def __init__(self, config: SDRConfig):
        if not _ADI_AVAILABLE:
            raise RuntimeError(
                "pyadi-iio 未安装。请运行: pip install pyadi-iio pylibiio"
            )
        super().__init__(config)
        uri = config.device_args or "ip:192.168.2.1"
        logger.info("连接 PlutoSDR: %s", uri)
        self._sdr = adi.Pluto(uri=uri)
        self.configure()

    def configure(self) -> None:
        cfg = self._config
        sdr = self._sdr

        # 采样率 (TX/RX 共用)
        sdr.sample_rate = int(cfg.sample_rate_hz)

        # 中心频率
        freq_hz = int(cfg.center_freq_hz)
        sdr.rx_lo = freq_hz
        sdr.tx_lo = freq_hz

        # 增益
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = cfg.rx_gain_db
        # Pluto TX 使用衰减值: 0 dB = 最大功率, -89.75 = 最小
        sdr.tx_hardwaregain_chan0 = self._gain_to_attn(cfg.tx_gain_db)

        # 带宽
        if cfg.bandwidth_hz > 0:
            sdr.rx_rf_bandwidth = int(cfg.bandwidth_hz)
            sdr.tx_rf_bandwidth = int(cfg.bandwidth_hz)
        else:
            sdr.rx_rf_bandwidth = int(cfg.sample_rate_hz)
            sdr.tx_rf_bandwidth = int(cfg.sample_rate_hz)

        # 缓冲区
        sdr.rx_buffer_size = cfg.rx_buffer_size
        sdr.tx_cyclic_buffer = False

        logger.info("PlutoSDR 配置完成: %s", self.status_string())

    @staticmethod
    def _gain_to_attn(gain_db: float) -> float:
        """将逻辑增益 (0=最小, 89.75=最大) 转为 Pluto TX 衰减值。"""
        attn = -max(0.0, min(89.75, gain_db))
        return max(PLUTO_TX_ATTN_MIN, min(PLUTO_TX_ATTN_MAX, attn))

    def set_frequency(self, freq_hz: float) -> None:
        freq_int = int(freq_hz)
        self._sdr.rx_lo = freq_int
        self._sdr.tx_lo = freq_int

    def set_sample_rate(self, rate_hz: float) -> None:
        self._sdr.sample_rate = int(rate_hz)
        self._config.sample_rate_hz = rate_hz

    def set_rx_gain(self, gain_db: float) -> None:
        self._sdr.rx_hardwaregain_chan0 = gain_db
        self._config.rx_gain_db = gain_db

    def set_tx_gain(self, gain_db: float) -> None:
        self._sdr.tx_hardwaregain_chan0 = self._gain_to_attn(gain_db)
        self._config.tx_gain_db = gain_db

    def set_bandwidth(self, bw_hz: float) -> None:
        bw_int = int(bw_hz) if bw_hz > 0 else int(self._config.sample_rate_hz)
        self._sdr.rx_rf_bandwidth = bw_int
        self._sdr.tx_rf_bandwidth = bw_int

    def transmit(self, samples: np.ndarray) -> int:
        # pyadi-iio 要求 int16 格式 (sc16) 或直接传 complex
        # adi.Pluto.tx() 接受 complex64/128 数组
        data = np.ascontiguousarray(samples.astype(np.complex64))
        self._sdr.tx(data)
        return len(data)

    def receive(self, num_samps: int) -> np.ndarray:
        # 调整缓冲区大小
        if self._sdr.rx_buffer_size != num_samps:
            self._sdr.rx_buffer_size = num_samps
        data = self._sdr.rx()
        return np.asarray(data, dtype=np.complex64)

    def close(self) -> None:
        # 停止 TX (发送零)
        self._sdr.tx_destroy_buffer()
        logger.info("PlutoSDR 已关闭")

    def status_string(self) -> str:
        cfg = self._config
        return (
            f"[Pluto] CH{cfg.channel_num} ({cfg.band}) "
            f"Fc={cfg.center_freq_hz / 1e6:.1f}MHz "
            f"Rate={cfg.sample_rate_hz / 1e6:.1f}Msps "
            f"RXG={cfg.rx_gain_db:.1f}dB TXG={cfg.tx_gain_db:.1f}dB"
        )

    @property
    def sdr(self):
        """获取底层 adi.Pluto 对象, 用于高级操作。"""
        return self._sdr
