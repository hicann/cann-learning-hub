"""UHD 后端 -- Ettus USRP 设备支持。

封装 UHD Python API, 实现 SDRDevice 接口。
保留对 USRP E310 等设备的完整支持。
"""

from __future__ import annotations

__all__ = [
    "UHDDevice",
    "uhd_available",
]

import logging

import numpy as np

from nearlink_sdr.phy.sdr_backend import SDRConfig, SDRDevice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UHD 条件导入
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


# E310 硬件常量
E310_FREQ_MIN_HZ = 70e6
E310_FREQ_MAX_HZ = 6e9
E310_BW_MAX_HZ = 56e6
E310_RX_GAIN_MAX = 76.0
E310_TX_GAIN_MAX = 89.75


class UHDDevice(SDRDevice):
    """USRP 设备接口 (基于 UHD)。"""

    def __init__(self, config: SDRConfig):
        if not _UHD_AVAILABLE:
            raise RuntimeError("UHD 未安装。请安装 uhd Python 绑定。")
        super().__init__(config)

        dev_args = config.device_args
        self._usrp = uhd.usrp.MultiUSRP(dev_args)
        logger.info("已连接 USRP: %s", self._usrp.get_mboard_name())

        # 流对象 (延迟创建)
        self._tx_streamer = None
        self._rx_streamer = None
        self._rx_buffer = None

        self.configure()

    def configure(self) -> None:
        cfg = self._config
        freq_hz = cfg.center_freq_hz
        usrp = self._usrp

        usrp.set_rx_rate(cfg.sample_rate_hz)
        usrp.set_tx_rate(cfg.sample_rate_hz)

        tune_req = uhd.libpyuhd.types.tune_request(freq_hz)
        usrp.set_rx_freq(tune_req, 0)
        usrp.set_tx_freq(tune_req, 0)

        usrp.set_rx_gain(cfg.rx_gain_db)
        usrp.set_tx_gain(cfg.tx_gain_db)

        if cfg.rx_antenna:
            usrp.set_rx_antenna(cfg.rx_antenna)
        if cfg.tx_antenna:
            usrp.set_tx_antenna(cfg.tx_antenna)

        if cfg.bandwidth_hz > 0:
            usrp.set_rx_bandwidth(cfg.bandwidth_hz)
            usrp.set_tx_bandwidth(cfg.bandwidth_hz)

        logger.info("USRP 配置完成: %s", self.status_string())

    def set_frequency(self, freq_hz: float) -> None:
        tune_req = uhd.libpyuhd.types.tune_request(freq_hz)
        self._usrp.set_rx_freq(tune_req, 0)
        self._usrp.set_tx_freq(tune_req, 0)

    def set_sample_rate(self, rate_hz: float) -> None:
        self._usrp.set_rx_rate(rate_hz)
        self._usrp.set_tx_rate(rate_hz)
        self._config.sample_rate_hz = rate_hz

    def set_rx_gain(self, gain_db: float) -> None:
        self._usrp.set_rx_gain(gain_db)
        self._config.rx_gain_db = gain_db

    def set_tx_gain(self, gain_db: float) -> None:
        self._usrp.set_tx_gain(gain_db)
        self._config.tx_gain_db = gain_db

    def set_bandwidth(self, bw_hz: float) -> None:
        if bw_hz > 0:
            self._usrp.set_rx_bandwidth(bw_hz)
            self._usrp.set_tx_bandwidth(bw_hz)

    def _ensure_tx_streamer(self):
        if self._tx_streamer is None:
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [0]
            self._tx_streamer = self._usrp.get_tx_stream(stream_args)

    def _ensure_rx_streamer(self, buf_size: int):
        if self._rx_streamer is None:
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [0]
            self._rx_streamer = self._usrp.get_rx_stream(stream_args)
            max_samps = self._rx_streamer.get_max_num_samps()
            actual = min(buf_size, max_samps)
            self._rx_buffer = np.zeros((1, actual), dtype=np.complex64)
            self._rx_metadata = uhd.types.RXMetadata()

    def transmit(self, samples: np.ndarray) -> int:
        self._ensure_tx_streamer()
        data = np.ascontiguousarray(samples.astype(np.complex64))
        metadata = uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = True
        return self._tx_streamer.send(data, metadata, self._config.stream_timeout_s)

    def receive(self, num_samps: int) -> np.ndarray:
        self._ensure_rx_streamer(min(num_samps, 4096))

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self._rx_streamer.issue_stream_cmd(stream_cmd)

        result = np.zeros(num_samps, dtype=np.complex64)
        collected = 0

        while collected < num_samps:
            n_recv = self._rx_streamer.recv(
                self._rx_buffer, self._rx_metadata, self._config.stream_timeout_s
            )
            if n_recv == 0:
                break
            n_copy = min(n_recv, num_samps - collected)
            result[collected : collected + n_copy] = self._rx_buffer[0, :n_copy]
            collected += n_copy

        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self._rx_streamer.issue_stream_cmd(stop_cmd)

        return result[:collected]

    def close(self) -> None:
        self._tx_streamer = None
        self._rx_streamer = None
        self._rx_buffer = None
        logger.info("UHD 设备已关闭")

    def status_string(self) -> str:
        cfg = self._config
        return (
            f"[UHD] {self._usrp.get_mboard_name()} "
            f"CH{cfg.channel_num} ({cfg.band}) "
            f"Fc={cfg.center_freq_hz / 1e6:.1f}MHz "
            f"Rate={cfg.sample_rate_hz / 1e6:.1f}Msps "
            f"RXG={cfg.rx_gain_db:.1f}dB TXG={cfg.tx_gain_db:.1f}dB"
        )
