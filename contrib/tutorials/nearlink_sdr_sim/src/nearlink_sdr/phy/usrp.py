"""USRP E310 硬件接口层。

封装 UHD Python API, 提供针对 SparkLink SLE 标准优化的 USRP 设备管理、
TX/RX 流接口以及与现有 PHY 模块的集成 Pipeline。

硬件参数 (USRP E310 / AD9361):
- 频率范围: 70 MHz - 6 GHz
- 最大带宽: 56 MHz
- ADC/DAC: 12-bit
- RX 增益范围: 0 - 76 dB
- TX 增益范围: 0 - 89.75 dB
- 天线端口: TX/RX, RX2
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CPU_FORMAT",
    "DEFAULT_WIRE_FORMAT",
    "E310_ADC_BITS",
    "E310_BW_MAX_HZ",
    "E310_FREQ_MAX_HZ",
    "E310_FREQ_MIN_HZ",
    "E310_RX_GAIN_MAX",
    "E310_RX_GAIN_MIN",
    "E310_TX_GAIN_MAX",
    "E310_TX_GAIN_MIN",
    "SLE_BANDWIDTHS_MHZ",
    "RXStream",
    "SLETransceiver",
    "TXStream",
    "USRPConfig",
    "USRPDevice",
    "uhd_available",
]


import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.freq_hopping import (
    BAND_2400,
    channel_to_freq,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UHD 条件导入 — 开发/测试环境可能不具备 UHD 驱动
# ---------------------------------------------------------------------------

try:
    import uhd
    _UHD_AVAILABLE = True
except ImportError:
    uhd = None  # type: ignore[assignment]
    _UHD_AVAILABLE = False


def uhd_available() -> bool:
    """检查当前环境是否安装了 UHD 驱动。"""
    return _UHD_AVAILABLE


# ---------------------------------------------------------------------------
# E310 硬件常量
# ---------------------------------------------------------------------------

E310_FREQ_MIN_HZ = 70e6
E310_FREQ_MAX_HZ = 6e9
E310_BW_MAX_HZ = 56e6
E310_RX_GAIN_MIN = 0.0
E310_RX_GAIN_MAX = 76.0
E310_TX_GAIN_MIN = 0.0
E310_TX_GAIN_MAX = 89.75
E310_ADC_BITS = 12

# SLE 标准支持的带宽 (MHz)
SLE_BANDWIDTHS_MHZ = (1.0, 2.0, 4.0)

# 默认 I/O 格式
DEFAULT_CPU_FORMAT = "fc32"   # np.complex64
DEFAULT_WIRE_FORMAT = "sc16"  # 16-bit signed complex on wire


# ---------------------------------------------------------------------------
# 设备配置
# ---------------------------------------------------------------------------

@dataclass
class USRPConfig:
    """USRP 设备配置参数。

    :ivar device_args: UHD 设备地址参数, 空字符串表示自动发现。
    :ivar channel_num: SLE 射频信道号。
    :ivar band: 频段标识 ("2400" / "5100" / "5800")。
    :ivar sample_rate_hz: 采样率 (Hz)。
    :ivar rx_gain_db: 接收增益 (dB)。
    :ivar tx_gain_db: 发射增益 (dB)。
    :ivar bandwidth_hz: 前端模拟滤波器带宽 (Hz), 0 表示不设置。
    :ivar rx_antenna: 接收天线端口名称。
    :ivar tx_antenna: 发射天线端口名称。
    :ivar clock_source: 参考时钟源。
    :ivar time_source: PPS 时间源。
    :ivar cpu_format: 主机端样本格式。
    :ivar wire_format: 线缆端样本格式。
    :ivar num_recv_frames: 接收缓冲区帧数, 增大可减少溢出。
    :ivar stream_timeout_s: 流操作超时时间 (秒)。
    """
    device_args: str = ""
    channel_num: int = 0
    band: str = BAND_2400
    sample_rate_hz: float = 1e6
    rx_gain_db: float = 30.0
    tx_gain_db: float = 20.0
    bandwidth_hz: float = 0.0
    rx_antenna: str = "RX2"
    tx_antenna: str = "TX/RX"
    clock_source: str = "internal"
    time_source: str = "internal"
    cpu_format: str = DEFAULT_CPU_FORMAT
    wire_format: str = DEFAULT_WIRE_FORMAT
    num_recv_frames: int = 512
    stream_timeout_s: float = 3.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        """校验配置参数有效性。"""
        freq_hz = self.center_freq_hz
        if not (E310_FREQ_MIN_HZ <= freq_hz <= E310_FREQ_MAX_HZ):
            raise ValueError(
                f"中心频率 {freq_hz / 1e6:.1f} MHz 超出 E310 支持范围 "
                f"[{E310_FREQ_MIN_HZ / 1e6:.0f}, {E310_FREQ_MAX_HZ / 1e6:.0f}] MHz"
            )
        if not (E310_RX_GAIN_MIN <= self.rx_gain_db <= E310_RX_GAIN_MAX):
            raise ValueError(
                f"RX gain {self.rx_gain_db} dB 超出范围 "
                f"[{E310_RX_GAIN_MIN}, {E310_RX_GAIN_MAX}]"
            )
        if not (E310_TX_GAIN_MIN <= self.tx_gain_db <= E310_TX_GAIN_MAX):
            raise ValueError(
                f"TX gain {self.tx_gain_db} dB 超出范围 "
                f"[{E310_TX_GAIN_MIN}, {E310_TX_GAIN_MAX}]"
            )
        if self.sample_rate_hz <= 0 or self.sample_rate_hz > E310_BW_MAX_HZ:
            raise ValueError(
                f"采样率 {self.sample_rate_hz / 1e6:.1f} MHz 超出范围 "
                f"(0, {E310_BW_MAX_HZ / 1e6:.0f}] MHz"
            )

    @property
    def center_freq_hz(self) -> float:
        """根据信道号和频段计算中心频率 (Hz)。"""
        return channel_to_freq(self.channel_num, self.band) * 1e6

    @property
    def bandwidth_mhz(self) -> float:
        """带宽 (MHz)。"""
        return self.sample_rate_hz / 1e6


# ---------------------------------------------------------------------------
# Mock USRP —— 用于无硬件环境下的开发与测试
# ---------------------------------------------------------------------------

class MockStreamCmd:
    """模拟 uhd.types.StreamCMD。"""
    def __init__(self, stream_mode):
        self.stream_mode = stream_mode
        self.stream_now = True
        self.num_samps = 0
        self.time_spec = None


class MockStreamMode:
    """模拟 uhd.types.StreamMode。"""
    start_cont = "start_cont"
    stop_cont = "stop_cont"
    num_done = "num_done"


class MockRXMetadata:
    """模拟 uhd.types.RXMetadata。"""
    def __init__(self):
        self.error_code = 0  # 0 = no error


class MockTXMetadata:
    """模拟 uhd.types.TXMetadata。"""
    def __init__(self):
        self.has_time_spec = False
        self.time_spec = None
        self.end_of_burst = False
        self.start_of_burst = False


class LoopbackBuffer:
    """TX→RX 环回缓冲区, 可选信道损伤。

    用于 MockUSRP 的 loopback 模式, 将 TX 发射的 IQ 样本经信道模型
    送入 RX 接收端, 实现无硬件的端到端仿真。
    """

    def __init__(self, channel_model: object | None = None, seed: int = 42):
        self._buffer: list[np.ndarray] = []
        self._channel = channel_model
        self._rng = np.random.default_rng(seed)

    def push(self, samples: np.ndarray):
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

    def clear(self):
        self._buffer.clear()


class MockStreamer:
    """模拟 UHD TX/RX Streamer。"""
    def __init__(
        self,
        direction: str = "rx",
        samps_per_buffer: int = 1000,
        loopback: LoopbackBuffer | None = None,
    ):
        self._direction = direction
        self._samps_per_buffer = samps_per_buffer
        self._running = False
        self._rng = np.random.default_rng(0)
        self._loopback = loopback

    def get_max_num_samps(self) -> int:
        return self._samps_per_buffer

    def issue_stream_cmd(self, cmd):
        if cmd.stream_mode == MockStreamMode.start_cont:
            self._running = True
        elif cmd.stream_mode == MockStreamMode.stop_cont:
            self._running = False

    def recv(self, buffer: np.ndarray, metadata, timeout: float = 3.0) -> int:
        """模拟接收: 环回模式返回 TX 发送的数据, 否则返回噪声。"""
        if not self._running:
            return 0
        n = min(buffer.shape[-1], self._samps_per_buffer)
        if self._loopback is not None and self._loopback.available > 0:
            data = self._loopback.pull(n)
            actual = len(data)
            if buffer.ndim == 2:
                buffer[0, :actual] = data
            else:
                buffer[:actual] = data
            return actual
        noise = (self._rng.standard_normal(n) +
                 1j * self._rng.standard_normal(n)).astype(np.complex64) * 0.01
        if buffer.ndim == 2:
            buffer[0, :n] = noise
        else:
            buffer[:n] = noise
        return n

    def send(self, buffer: np.ndarray, metadata, timeout: float = 3.0) -> int:
        """模拟发射: 环回模式写入缓冲区, 否则直接消耗。"""
        data = buffer[0] if buffer.ndim == 2 else buffer
        if self._loopback is not None:
            self._loopback.push(data)
        return len(data)


class MockUSRP:
    """模拟 uhd.usrp.MultiUSRP, 用于无硬件测试。

    记录所有配置调用的参数, 并提供可验证的状态。
    支持 loopback 模式: TX 发射的数据经可选信道模型后送入 RX。
    """

    def __init__(self, args: str = "", loopback: LoopbackBuffer | None = None):
        self._args = args
        self._rx_rate = 1e6
        self._tx_rate = 1e6
        self._rx_freq = 2.402e9
        self._tx_freq = 2.402e9
        self._rx_gain = 30.0
        self._tx_gain = 20.0
        self._rx_bw = 0.0
        self._tx_bw = 0.0
        self._rx_antenna = "RX2"
        self._tx_antenna = "TX/RX"
        self._clock_source = "internal"
        self._time_source = "internal"
        self._call_log: list[tuple[str, dict]] = []
        self._loopback = loopback

    def _log(self, method: str, **kwargs):
        self._call_log.append((method, kwargs))

    def set_rx_rate(self, rate: float, chan: int = 0):
        self._log("set_rx_rate", rate=rate, chan=chan)
        self._rx_rate = rate

    def get_rx_rate(self, chan: int = 0) -> float:
        return self._rx_rate

    def set_tx_rate(self, rate: float, chan: int = 0):
        self._log("set_tx_rate", rate=rate, chan=chan)
        self._tx_rate = rate

    def get_tx_rate(self, chan: int = 0) -> float:
        return self._tx_rate

    def set_rx_freq(self, tune_request, chan: int = 0):
        self._log("set_rx_freq", freq=tune_request, chan=chan)
        if isinstance(tune_request, (int, float)):
            self._rx_freq = float(tune_request)
        else:
            self._rx_freq = float(tune_request)

    def get_rx_freq(self, chan: int = 0) -> float:
        return self._rx_freq

    def set_tx_freq(self, tune_request, chan: int = 0):
        self._log("set_tx_freq", freq=tune_request, chan=chan)
        if isinstance(tune_request, (int, float)):
            self._tx_freq = float(tune_request)
        else:
            self._tx_freq = float(tune_request)

    def get_tx_freq(self, chan: int = 0) -> float:
        return self._tx_freq

    def set_rx_gain(self, gain: float, chan: int = 0):
        self._log("set_rx_gain", gain=gain, chan=chan)
        self._rx_gain = gain

    def get_rx_gain(self, chan: int = 0) -> float:
        return self._rx_gain

    def set_tx_gain(self, gain: float, chan: int = 0):
        self._log("set_tx_gain", gain=gain, chan=chan)
        self._tx_gain = gain

    def get_tx_gain(self, chan: int = 0) -> float:
        return self._tx_gain

    def set_rx_bandwidth(self, bw: float, chan: int = 0):
        self._log("set_rx_bandwidth", bw=bw, chan=chan)
        self._rx_bw = bw

    def get_rx_bandwidth(self, chan: int = 0) -> float:
        return self._rx_bw

    def set_tx_bandwidth(self, bw: float, chan: int = 0):
        self._log("set_tx_bandwidth", bw=bw, chan=chan)
        self._tx_bw = bw

    def get_tx_bandwidth(self, chan: int = 0) -> float:
        return self._tx_bw

    def set_rx_antenna(self, ant: str, chan: int = 0):
        self._log("set_rx_antenna", ant=ant, chan=chan)
        self._rx_antenna = ant

    def get_rx_antenna(self, chan: int = 0) -> str:
        return self._rx_antenna

    def set_tx_antenna(self, ant: str, chan: int = 0):
        self._log("set_tx_antenna", ant=ant, chan=chan)
        self._tx_antenna = ant

    def get_tx_antenna(self, chan: int = 0) -> str:
        return self._tx_antenna

    def set_clock_source(self, source: str, mboard: int = 0):
        self._log("set_clock_source", source=source, mboard=mboard)
        self._clock_source = source

    def set_time_source(self, source: str, mboard: int = 0):
        self._log("set_time_source", source=source, mboard=mboard)
        self._time_source = source

    def get_mboard_name(self, mboard: int = 0) -> str:
        return "E310 (Mock)"

    def get_rx_stream(self, stream_args) -> MockStreamer:
        self._log("get_rx_stream", args=str(stream_args))
        return MockStreamer("rx", loopback=self._loopback)

    def get_tx_stream(self, stream_args) -> MockStreamer:
        self._log("get_tx_stream", args=str(stream_args))
        return MockStreamer("tx", loopback=self._loopback)

    def get_pp_string(self) -> str:
        return (
            f"MockUSRP E310\n"
            f"  RX: freq={self._rx_freq / 1e6:.1f} MHz, "
            f"rate={self._rx_rate / 1e6:.1f} Msps, "
            f"gain={self._rx_gain:.1f} dB\n"
            f"  TX: freq={self._tx_freq / 1e6:.1f} MHz, "
            f"rate={self._tx_rate / 1e6:.1f} Msps, "
            f"gain={self._tx_gain:.1f} dB"
        )


# ---------------------------------------------------------------------------
# USRP 设备管理
# ---------------------------------------------------------------------------

class USRPDevice:
    """USRP E310 设备管理。

    封装 UHD MultiUSRP 对象, 提供 SLE 标准所需的设备配置和控制接口。
    无硬件时自动使用 MockUSRP。
    """

    def __init__(
        self,
        config: USRPConfig | None = None,
        use_mock: bool = False,
        loopback: LoopbackBuffer | None = None,
    ):
        """初始化 USRP 设备。

        :param config: 设备配置, None 时使用默认配置。
        :param use_mock: 强制使用 MockUSRP。
        :param loopback: 环回缓冲区, 仅 Mock 模式有效。
        """
        self._config = config or USRPConfig()
        self._use_mock = use_mock or not _UHD_AVAILABLE

        if self._use_mock:
            self._usrp = MockUSRP(self._config.device_args, loopback=loopback)
            logger.info("使用 MockUSRP (无硬件模式)")
        else:
            dev_args = self._config.device_args
            if self._config.num_recv_frames > 0:
                sep = "," if dev_args else ""
                dev_args += f"{sep}num_recv_frames={self._config.num_recv_frames}"
            self._usrp = uhd.usrp.MultiUSRP(dev_args)
            logger.info("已连接 USRP: %s", self._usrp.get_mboard_name())

        self._configure()

    def _configure(self):
        """将 USRPConfig 的参数应用到设备。"""
        cfg = self._config
        freq_hz = cfg.center_freq_hz

        # 时钟与同步源
        self._usrp.set_clock_source(cfg.clock_source)
        self._usrp.set_time_source(cfg.time_source)

        # RX 配置
        self._usrp.set_rx_rate(cfg.sample_rate_hz)
        self._set_freq("rx", freq_hz)
        self._usrp.set_rx_gain(cfg.rx_gain_db)
        self._usrp.set_rx_antenna(cfg.rx_antenna)
        if cfg.bandwidth_hz > 0:
            self._usrp.set_rx_bandwidth(cfg.bandwidth_hz)

        # TX 配置
        self._usrp.set_tx_rate(cfg.sample_rate_hz)
        self._set_freq("tx", freq_hz)
        self._usrp.set_tx_gain(cfg.tx_gain_db)
        self._usrp.set_tx_antenna(cfg.tx_antenna)
        if cfg.bandwidth_hz > 0:
            self._usrp.set_tx_bandwidth(cfg.bandwidth_hz)

        logger.info("设备配置完成: %s", self.status_string())

    def _set_freq(self, direction: str, freq_hz: float, chan: int = 0):
        """设置中心频率。"""
        if self._use_mock:
            if direction == "rx":
                self._usrp.set_rx_freq(freq_hz, chan)
            else:
                self._usrp.set_tx_freq(freq_hz, chan)
        else:
            tune_req = uhd.libpyuhd.types.tune_request(freq_hz)
            if direction == "rx":
                self._usrp.set_rx_freq(tune_req, chan)
            else:
                self._usrp.set_tx_freq(tune_req, chan)

    def tune_channel(self, channel_num: int, band: str | None = None):
        """切换到指定 SLE 射频信道。

        :param channel_num: 射频信道号。
        :param band: 频段标识, None 沿用当前配置。
        """
        if band is not None:
            self._config.band = band
        self._config.channel_num = channel_num
        freq_hz = self._config.center_freq_hz
        self._set_freq("rx", freq_hz)
        self._set_freq("tx", freq_hz)
        logger.debug("切换至信道 %d, 频率 %.1f MHz", channel_num, freq_hz / 1e6)

    def set_rx_gain(self, gain_db: float):
        """设置 RX 增益。"""
        if not (E310_RX_GAIN_MIN <= gain_db <= E310_RX_GAIN_MAX):
            raise ValueError(f"RX gain {gain_db} dB 超出范围")
        self._config.rx_gain_db = gain_db
        self._usrp.set_rx_gain(gain_db)

    def set_tx_gain(self, gain_db: float):
        """设置 TX 增益。"""
        if not (E310_TX_GAIN_MIN <= gain_db <= E310_TX_GAIN_MAX):
            raise ValueError(f"TX gain {gain_db} dB 超出范围")
        self._config.tx_gain_db = gain_db
        self._usrp.set_tx_gain(gain_db)

    def set_sample_rate(self, rate_hz: float):
        """设置收发采样率。"""
        if rate_hz <= 0 or rate_hz > E310_BW_MAX_HZ:
            raise ValueError(f"采样率 {rate_hz / 1e6:.1f} MHz 超出范围")
        self._config.sample_rate_hz = rate_hz
        self._usrp.set_rx_rate(rate_hz)
        self._usrp.set_tx_rate(rate_hz)

    @property
    def config(self) -> USRPConfig:
        return self._config

    @property
    def usrp(self):
        """获取底层 UHD/Mock 设备对象。"""
        return self._usrp

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    def status_string(self) -> str:
        """返回可读的设备状态摘要。"""
        cfg = self._config
        return (
            f"[{'Mock' if self._use_mock else 'HW'}] "
            f"CH{cfg.channel_num} ({cfg.band}) "
            f"Fc={cfg.center_freq_hz / 1e6:.1f}MHz "
            f"Rate={cfg.sample_rate_hz / 1e6:.1f}Msps "
            f"RXG={cfg.rx_gain_db:.1f}dB TXG={cfg.tx_gain_db:.1f}dB"
        )


# ---------------------------------------------------------------------------
# TX 发射流
# ---------------------------------------------------------------------------

class TXStream:
    """USRP 发射流接口。

    管理 TX streamer 的生命周期, 提供帧级发射方法。
    """

    def __init__(self, device: USRPDevice):
        self._device = device
        self._streamer = None
        self._metadata = None

    def open(self):
        """创建 TX streamer。"""
        if self._device.is_mock:
            stream_args = "fc32"
        else:
            stream_args = uhd.usrp.StreamArgs(
                self._device.config.cpu_format,
                self._device.config.wire_format,
            )
            stream_args.channels = [0]
        self._streamer = self._device.usrp.get_tx_stream(stream_args)
        if self._device.is_mock:
            self._metadata = MockTXMetadata()
        else:
            self._metadata = uhd.types.TXMetadata()
        logger.debug("TX streamer 已创建")

    def send(self, samples: np.ndarray) -> int:
        """发射一帧 IQ 样本。

        :param samples: 复基带样本, dtype=complex64, shape (N,)。

        :returns: 实际发射的样本数。
        """
        if self._streamer is None:
            raise RuntimeError("TX streamer 未打开, 请先调用 open()")

        samples = np.ascontiguousarray(samples.astype(np.complex64))
        self._metadata.start_of_burst = True
        self._metadata.end_of_burst = True

        n_sent = self._streamer.send(
            samples, self._metadata, self._device.config.stream_timeout_s
        )
        return n_sent

    def send_continuous(self, samples: np.ndarray, num_repeats: int = 1) -> int:
        """连续发射, 重复指定次数。

        :param samples: 单帧 IQ 样本。
        :param num_repeats: 重复次数。

        :returns: 总发射样本数。
        """
        if self._streamer is None:
            raise RuntimeError("TX streamer 未打开, 请先调用 open()")

        samples = np.ascontiguousarray(samples.astype(np.complex64))
        total = 0
        for i in range(num_repeats):
            self._metadata.start_of_burst = (i == 0)
            self._metadata.end_of_burst = (i == num_repeats - 1)
            n = self._streamer.send(
                samples, self._metadata, self._device.config.stream_timeout_s
            )
            total += n
        return total

    def close(self):
        """关闭 TX streamer。"""
        self._streamer = None
        self._metadata = None
        logger.debug("TX streamer 已关闭")


# ---------------------------------------------------------------------------
# RX 接收流
# ---------------------------------------------------------------------------

class RXStream:
    """USRP 接收流接口。

    管理 RX streamer 的生命周期, 提供定量接收和连续接收方法。
    """

    def __init__(self, device: USRPDevice):
        self._device = device
        self._streamer = None
        self._metadata = None
        self._recv_buffer = None
        self._running = False

    def open(self, samps_per_buffer: int = 1000):
        """创建 RX streamer 并分配接收缓冲区。

        :param samps_per_buffer: 每次 recv 调用的缓冲区大小。
        """
        if self._device.is_mock:
            stream_args = "fc32"
        else:
            stream_args = uhd.usrp.StreamArgs(
                self._device.config.cpu_format,
                self._device.config.wire_format,
            )
            stream_args.channels = [0]
        self._streamer = self._device.usrp.get_rx_stream(stream_args)
        if self._device.is_mock:
            self._metadata = MockRXMetadata()
        else:
            self._metadata = uhd.types.RXMetadata()
        max_samps = self._streamer.get_max_num_samps()
        buf_size = min(samps_per_buffer, max_samps)
        self._recv_buffer = np.zeros((1, buf_size), dtype=np.complex64)
        logger.debug("RX streamer 已创建, 缓冲区 %d 样本", buf_size)

    def recv_num_samps(self, num_samps: int) -> np.ndarray:
        """接收指定数量的样本。

        :param num_samps: 要接收的样本数。

        :returns: 复基带样本数组, shape (num_samps,), dtype=complex64。
        """
        if self._streamer is None:
            raise RuntimeError("RX streamer 未打开, 请先调用 open()")

        # 开始连续流
        if self._device.is_mock:
            stream_cmd = MockStreamCmd(MockStreamMode.start_cont)
        else:
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self._streamer.issue_stream_cmd(stream_cmd)

        samples = np.zeros(num_samps, dtype=np.complex64)
        collected = 0

        while collected < num_samps:
            n_recv = self._streamer.recv(
                self._recv_buffer, self._metadata,
                self._device.config.stream_timeout_s,
            )
            if n_recv == 0:
                break
            n_copy = min(n_recv, num_samps - collected)
            samples[collected:collected + n_copy] = self._recv_buffer[0, :n_copy]
            collected += n_copy

        # 停止流
        if self._device.is_mock:
            stop_cmd = MockStreamCmd(MockStreamMode.stop_cont)
        else:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self._streamer.issue_stream_cmd(stop_cmd)

        return samples[:collected]

    def start_continuous(self):
        """启动连续接收模式。"""
        if self._streamer is None:
            raise RuntimeError("RX streamer 未打开, 请先调用 open()")
        if self._device.is_mock:
            stream_cmd = MockStreamCmd(MockStreamMode.start_cont)
        else:
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self._streamer.issue_stream_cmd(stream_cmd)
        self._running = True

    def recv_once(self) -> np.ndarray:
        """在连续模式下接收一个缓冲区的样本。

        :returns: 接收到的样本, 长度可能小于缓冲区大小。
        """
        if not self._running:
            raise RuntimeError("连续接收未启动, 请先调用 start_continuous()")
        n = self._streamer.recv(
            self._recv_buffer, self._metadata,
            self._device.config.stream_timeout_s,
        )
        return self._recv_buffer[0, :n].copy()

    def stop_continuous(self):
        """停止连续接收。"""
        if self._running:
            if self._device.is_mock:
                stop_cmd = MockStreamCmd(MockStreamMode.stop_cont)
            else:
                stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            self._streamer.issue_stream_cmd(stop_cmd)
            self._running = False

    def close(self):
        """关闭 RX streamer。"""
        self.stop_continuous()
        self._streamer = None
        self._metadata = None
        self._recv_buffer = None
        logger.debug("RX streamer 已关闭")


# ---------------------------------------------------------------------------
# SLE 收发 Pipeline
# ---------------------------------------------------------------------------

class SLETransceiver:
    """SparkLink SLE 实时收发 Pipeline。

    将 USRP 设备与 PHY 调制/解调模块集成, 提供帧级收发功能。
    支持跳频操作。
    """

    def __init__(self, device: USRPDevice):
        self._device = device
        self._tx_stream = TXStream(device)
        self._rx_stream = RXStream(device)
        self._is_open = False

    def open(self, rx_buf_size: int = 1000):
        """初始化收发流。"""
        self._tx_stream.open()
        self._rx_stream.open(samps_per_buffer=rx_buf_size)
        self._is_open = True
        logger.info("SLE 收发 Pipeline 已就绪")

    def transmit_iq(self, iq_samples: np.ndarray) -> int:
        """发射预调制的 IQ 样本。

        :param iq_samples: 复基带信号。

        :returns: 实际发射的样本数。
        """
        if not self._is_open:
            raise RuntimeError("Pipeline 未打开")
        return self._tx_stream.send(iq_samples)

    def receive_iq(self, num_samps: int) -> np.ndarray:
        """接收指定数量的 IQ 样本。

        :param num_samps: 要接收的样本数。

        :returns: 接收到的复基带信号。
        """
        if not self._is_open:
            raise RuntimeError("Pipeline 未打开")
        return self._rx_stream.recv_num_samps(num_samps)

    def transmit_frame(
        self,
        frame_bits: np.ndarray,
        modulate_fn: Callable[[np.ndarray], np.ndarray],
    ) -> int:
        """调制并发射一帧。

        :param frame_bits: 帧比特序列。
        :param modulate_fn: 调制函数, 接收比特数组, 返回 IQ 样本。

        :returns: 实际发射的样本数。
        """
        iq = modulate_fn(frame_bits)
        return self.transmit_iq(iq)

    def receive_frame(
        self,
        num_samps: int,
        demodulate_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """接收并解调一帧。

        :param num_samps: 接收样本数。
        :param demodulate_fn: 解调函数, 接收 IQ 样本, 返回比特数组。

        :returns: 解调后的比特序列。
        """
        iq = self.receive_iq(num_samps)
        return demodulate_fn(iq)

    def hop_and_transmit(
        self,
        channel_num: int,
        iq_samples: np.ndarray,
        band: str | None = None,
    ) -> int:
        """跳频后发射。

        :param channel_num: 目标信道号。
        :param iq_samples: 要发射的 IQ 样本。
        :param band: 频段标识, None 沿用当前配置。

        :returns: 实际发射的样本数。
        """
        self._device.tune_channel(channel_num, band)
        return self.transmit_iq(iq_samples)

    def hop_and_receive(
        self,
        channel_num: int,
        num_samps: int,
        band: str | None = None,
    ) -> np.ndarray:
        """跳频后接收。

        :param channel_num: 目标信道号。
        :param num_samps: 接收样本数。
        :param band: 频段标识, None 沿用当前配置。

        :returns: 接收到的 IQ 样本。
        """
        self._device.tune_channel(channel_num, band)
        return self.receive_iq(num_samps)

    def close(self):
        """关闭所有收发流。"""
        self._tx_stream.close()
        self._rx_stream.close()
        self._is_open = False
        logger.info("SLE 收发 Pipeline 已关闭")

    @property
    def device(self) -> USRPDevice:
        return self._device

    @property
    def is_open(self) -> bool:
        return self._is_open
