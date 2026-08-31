"""SDR 硬件后端抽象层。

定义与具体 SDR 硬件无关的统一接口, 支持 UHD (USRP), pyadi-iio (PlutoSDR/ANTSDR),
Mock (无硬件仿真) 三种后端。所有后端实现 SDRDevice 接口后即可接入 SLETransceiver。
"""

from __future__ import annotations

__all__ = [
    "SDRConfig",
    "SDRDevice",
    "create_device",
]

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.freq_hopping import (
    BAND_2400,
    channel_to_freq,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 通用 SDR 配置
# ---------------------------------------------------------------------------

@dataclass
class SDRConfig:
    """SDR 设备配置参数 (后端无关)。

    :ivar backend: 后端类型 ("uhd" / "pluto" / "mock")。
    :ivar device_args: 设备地址参数。UHD: 如 "addr=192.168.10.2"; Pluto: 如 "ip:192.168.2.1"。
    :ivar channel_num: SLE 射频信道号。
    :ivar band: 频段标识 ("2400" / "5100" / "5800")。
    :ivar sample_rate_hz: 采样率 (Hz)。
    :ivar rx_gain_db: 接收增益 (dB)。
    :ivar tx_gain_db: 发射增益 (dB)。
    :ivar bandwidth_hz: 前端模拟滤波器带宽 (Hz), 0 表示自动。
    :ivar rx_antenna: 接收天线端口名称。
    :ivar tx_antenna: 发射天线端口名称。
    :ivar stream_timeout_s: 流操作超时时间 (秒)。
    :ivar rx_buffer_size: 接收缓冲区大小 (样本数)。
    """

    backend: str = "mock"
    device_args: str = ""
    channel_num: int = 0
    band: str = BAND_2400
    sample_rate_hz: float = 1e6
    rx_gain_db: float = 30.0
    tx_gain_db: float = 20.0
    bandwidth_hz: float = 0.0
    rx_antenna: str = ""
    tx_antenna: str = ""
    stream_timeout_s: float = 3.0
    rx_buffer_size: int = 4096

    @property
    def center_freq_hz(self) -> float:
        """根据信道号和频段计算中心频率 (Hz)。"""
        return channel_to_freq(self.channel_num, self.band) * 1e6


# ---------------------------------------------------------------------------
# SDR 设备抽象接口
# ---------------------------------------------------------------------------

class SDRDevice(ABC):
    """SDR 设备抽象基类。

    所有后端 (UHD, Pluto, Mock) 必须实现此接口。
    """

    def __init__(self, config: SDRConfig):
        self._config = config

    @property
    def config(self) -> SDRConfig:
        return self._config

    @abstractmethod
    def configure(self) -> None:
        """将配置参数应用到设备。"""

    @abstractmethod
    def set_frequency(self, freq_hz: float) -> None:
        """设置 TX/RX 中心频率。"""

    @abstractmethod
    def set_sample_rate(self, rate_hz: float) -> None:
        """设置收发采样率。"""

    @abstractmethod
    def set_rx_gain(self, gain_db: float) -> None:
        """设置接收增益。"""

    @abstractmethod
    def set_tx_gain(self, gain_db: float) -> None:
        """设置发射增益。"""

    @abstractmethod
    def set_bandwidth(self, bw_hz: float) -> None:
        """设置前端模拟滤波器带宽。"""

    def tune_channel(self, channel_num: int, band: str | None = None) -> None:
        """切换到指定 SLE 射频信道。

        :param channel_num: 射频信道号。
        :param band: 频段标识, None 沿用当前配置。
        """
        if band is not None:
            self._config.band = band
        self._config.channel_num = channel_num
        freq_hz = self._config.center_freq_hz
        self.set_frequency(freq_hz)
        logger.debug("切换至信道 %d, 频率 %.1f MHz", channel_num, freq_hz / 1e6)

    @abstractmethod
    def transmit(self, samples: np.ndarray) -> int:
        """发射 IQ 样本。

        :param samples: 复基带样本, dtype=complex64。

        :returns: 实际发射的样本数。
        """

    @abstractmethod
    def receive(self, num_samps: int) -> np.ndarray:
        """接收指定数量的 IQ 样本。

        :param num_samps: 要接收的样本数。

        :returns: 复基带样本数组, dtype=complex64。
        """

    @abstractmethod
    def close(self) -> None:
        """释放设备资源。"""

    @abstractmethod
    def status_string(self) -> str:
        """返回可读的设备状态摘要。"""

    @property
    def is_mock(self) -> bool:
        """是否为无硬件模拟模式。"""
        return self._config.backend == "mock"


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_device(config: SDRConfig) -> SDRDevice:
    """根据配置创建对应的 SDR 设备实例。

    :param config: SDR 配置, 通过 backend 字段选择后端。

    :returns: SDRDevice 实例。
    """
    backend = config.backend.lower()

    if backend == "mock":
        from nearlink_sdr.phy.mock_backend import MockDevice
        return MockDevice(config)

    if backend == "uhd":
        from nearlink_sdr.phy.uhd_backend import UHDDevice
        return UHDDevice(config)

    if backend == "pluto":
        from nearlink_sdr.phy.pluto_backend import PlutoDevice
        return PlutoDevice(config)

    raise ValueError(f"不支持的后端类型: {backend!r}, 可选: mock, uhd, pluto")
