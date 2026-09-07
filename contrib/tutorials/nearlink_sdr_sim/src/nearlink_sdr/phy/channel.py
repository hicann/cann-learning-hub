"""TXS-10002-2025 信道模型: AWGN / Rayleigh / Rician / 多径频率选择性衰落 / Doppler 时变 / 多用户干扰。

SparkLink SLE 工作于 2.4 GHz ISM 频段,典型场景为室内短距通信。
参考 IEEE 802.15.4 / ITU-R P.1238 室内信道参数:
  - RMS 时延扩展: 10 ~ 50 ns
  - Rician K 因子: 3 ~ 10 dB (视距)
  - 最大多普勒频移: < 50 Hz (行人速度)

Doppler 时变信道采用 Jakes 求和正弦模型, 生成具有经典 U 形功率谱密度的
时间相关衰落序列, 自相关函数为 J₀(2π·f_d·Δt)。

多用户干扰模型支持同信道干扰和邻信道干扰, 基于 TXS-10002-2025 标准
第 8.3.2 条接收机选择性要求。
"""


__all__ = [
    "PDP_2TAP",
    "PDP_INDOOR_OFFICE",
    "ChannelConfig",
    "ChannelModel",
    "InterferenceConfig",
    "add_interference",
    "compute_sinr",
]


from dataclasses import dataclass, field

import numpy as np
from scipy.special import j0 as bessel_j0

# ── 典型室内功率时延谱 (PDP) ──

# 简化两径模型: 直射径 + 一条反射径
PDP_2TAP = [
    (0, 0.0),       # 直射径, 0 dB
    (1, -10.0),      # 反射径, -10 dB, 延迟 1 符号
]

# ITU Indoor Office B (简化 6 径模型, 相对时延单位: 符号周期)
PDP_INDOOR_OFFICE = [
    (0, 0.0),
    (1, -3.6),
    (2, -6.5),
    (3, -9.4),
    (4, -12.4),
    (5, -15.4),
]


@dataclass
class ChannelConfig:
    """信道配置参数。

    :ivar snr_db: 信噪比 (dB)。
    :ivar channel_type: "awgn" | "rayleigh" | "rician" | "multipath"。
    :ivar rician_k_db: Rician K 因子 (dB), 仅 "rician" 模式使用。
    :ivar pdp: 功率时延谱 [(delay_samples, power_dB), ...], 仅 "multipath" 使用。
    :ivar max_doppler_hz: 最大多普勒频移 (Hz)。0 表示准静态。
    :ivar symbol_rate_hz: 符号速率 (Hz), 用于计算归一化 Doppler 频移。
    :ivar n_sinusoids: Jakes 模型正弦分量数, 值越大频谱越精确。
    :ivar seed: 随机种子。
    """
    snr_db: float = 10.0
    channel_type: str = "awgn"
    rician_k_db: float = 6.0
    pdp: list[tuple[int, float]] = field(default_factory=lambda: list(PDP_2TAP))
    max_doppler_hz: float = 0.0
    symbol_rate_hz: float = 1e6
    n_sinusoids: int = 16
    seed: int | None = None


class ChannelModel:
    """多模信道模型,支持 AWGN / 平坦衰落 / 频率选择性衰落。"""

    def __init__(self, snr_db: float = 10.0, config: ChannelConfig | None = None):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = ChannelConfig(snr_db=snr_db)
        self._rng = np.random.default_rng(self.cfg.seed)
        self._last_taps: np.ndarray | None = None

    # ── 公共接口 ──

    def apply_awgn(self, signal: np.ndarray, sps: int = 1) -> np.ndarray:
        """仅加 AWGN 噪声。

        :param sps: 每符号采样数, 用于补偿上采样/RRC 摊薄的功率。默认 1。
        """
        return self._add_noise(signal, self.cfg.snr_db, sps)

    def apply_fading(self, signal: np.ndarray, sps: int = 1) -> np.ndarray:
        """按 config 类型应用衰落 + AWGN, 同时缓存信道系数到 last_taps。

        :param sps: 每符号采样数, 用于补偿上采样/RRC 摊薄的功率。默认 1。
        """
        ct = self.cfg.channel_type
        n = len(signal)
        if ct == "awgn":
            self._last_taps = np.ones((1, n), dtype=complex)
            return self._add_noise(signal, self.cfg.snr_db, sps)
        elif ct == "rayleigh":
            h = self._gen_rayleigh_coeffs(n)
            self._last_taps = h.reshape(1, -1)
            return self._add_noise(signal * h, self.cfg.snr_db, sps)
        elif ct == "rician":
            h = self._gen_rician_coeffs(n)
            self._last_taps = h.reshape(1, -1)
            return self._add_noise(signal * h, self.cfg.snr_db, sps)
        elif ct == "multipath":
            taps = self._gen_multipath_taps(n)
            self._last_taps = taps
            faded = self._apply_multipath_taps(signal, taps)
            return self._add_noise(faded, self.cfg.snr_db, sps)
        else:
            raise ValueError(f"Unknown channel type: {ct}")

    @property
    def last_taps(self) -> np.ndarray | None:
        """上一次 apply_fading 使用的信道系数, 供均衡器使用。"""
        return self._last_taps

    def get_channel_taps(self, n_symbols: int) -> np.ndarray:
        """返回信道抽头系数矩阵 (n_taps × n_symbols), 用于均衡器。

        对于平坦衰落: 返回 (1, n_symbols)。
        对于多径: 返回 (max_delay+1, n_symbols)。
        """
        ct = self.cfg.channel_type
        if ct == "awgn":
            return np.ones((1, n_symbols), dtype=complex)
        elif ct == "rayleigh":
            return self._gen_rayleigh_coeffs(n_symbols).reshape(1, -1)
        elif ct == "rician":
            return self._gen_rician_coeffs(n_symbols).reshape(1, -1)
        elif ct == "multipath":
            return self._gen_multipath_taps(n_symbols)
        else:
            raise ValueError(f"Unknown channel type: {ct}")

    @property
    def noise_variance(self) -> float:
        """AWGN 噪声方差 (单边, 假设单位信号功率)。"""
        snr_lin = 10.0 ** (self.cfg.snr_db / 10.0)
        return 1.0 / (2.0 * snr_lin) if snr_lin > 0 else 1e10

    # ── 平坦衰落 ──

    def _rayleigh_flat(self, signal: np.ndarray) -> np.ndarray:
        h = self._gen_rayleigh_coeffs(len(signal))
        self._last_taps = h.reshape(1, -1)
        return signal * h

    def _rician_flat(self, signal: np.ndarray) -> np.ndarray:
        h = self._gen_rician_coeffs(len(signal))
        self._last_taps = h.reshape(1, -1)
        return signal * h

    def _gen_rayleigh_coeffs(self, n: int) -> np.ndarray:
        """生成 Rayleigh 衰落系数。

        max_doppler_hz > 0 时使用 Jakes 求和正弦模型, 产生具有经典 U 形
        功率谱密度的时间相关衰落; 否则为准静态 (整段同一系数)。
        """
        if self.cfg.max_doppler_hz <= 0:
            h = (self._rng.standard_normal() + 1j * self._rng.standard_normal()) / np.sqrt(2)
            return np.full(n, h)
        return self._jakes_fading(n)

    def _gen_rician_coeffs(self, n: int) -> np.ndarray:
        """生成 Rician 衰落系数。K = LOS功率 / 散射功率。"""
        k_lin = 10.0 ** (self.cfg.rician_k_db / 10.0)
        los_amp = np.sqrt(k_lin / (k_lin + 1.0))
        scatter_amp = np.sqrt(1.0 / (k_lin + 1.0))

        if self.cfg.max_doppler_hz <= 0:
            scatter = (self._rng.standard_normal() + 1j * self._rng.standard_normal()) / np.sqrt(2)
            h = los_amp + scatter_amp * scatter
            return np.full(n, h)
        scatter = self._jakes_fading(n)
        return los_amp + scatter_amp * scatter

    # ── Jakes 求和正弦模型 ──

    def _jakes_fading(self, n: int) -> np.ndarray:
        """Jakes 求和正弦法生成时间相关衰落序列。

        使用 N_osc 个正弦分量叠加, 到达角均匀分布, 初始相位随机。
        输出自相关逼近 J₀(2π·f_d·Δt)。

        参考: W.C. Jakes, "Microwave Mobile Communications", 1974
        """
        fd = self.cfg.max_doppler_hz
        fs = self.cfg.symbol_rate_hz
        n_osc = self.cfg.n_sinusoids

        t = np.arange(n, dtype=np.float64) / fs

        # 随机初始相位
        phi = self._rng.uniform(0, 2 * np.pi, n_osc)
        theta = self._rng.uniform(0, 2 * np.pi, n_osc)

        # 到达角均匀分布
        alpha = (2 * np.pi * np.arange(1, n_osc + 1) + phi) / (4 * n_osc)

        # I/Q 两路正弦叠加
        h_i = np.zeros(n, dtype=np.float64)
        h_q = np.zeros(n, dtype=np.float64)
        for k in range(n_osc):
            cos_alpha = np.cos(alpha[k])
            w_d_cos = 2 * np.pi * fd * cos_alpha
            h_i += np.cos(w_d_cos * t + theta[k])
            h_q += np.sin(w_d_cos * t + theta[k])

        scale = 1.0 / np.sqrt(n_osc)
        return (h_i + 1j * h_q) * scale

    # ── 多径频率选择性衰落 ──

    def _multipath(self, signal: np.ndarray) -> np.ndarray:
        """应用多径信道 (FIR 卷积模型)。"""
        taps = self._gen_multipath_taps(len(signal))
        self._last_taps = taps
        return self._apply_multipath_taps(signal, taps)

    @staticmethod
    def _apply_multipath_taps(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
        """用给定的抽头系数对信号做 FIR 多径卷积。"""
        n_taps = taps.shape[0]
        n = len(signal)
        out = np.zeros(n, dtype=complex)
        for k in range(n_taps):
            delay = k
            if delay < n:
                shifted = np.zeros(n, dtype=complex)
                shifted[delay:] = signal[:n - delay]
                out += taps[k, :n] * shifted
        return out

    def _gen_multipath_taps(self, n: int) -> np.ndarray:
        """生成多径信道抽头系数。

        返回 (n_taps, n) 的复数矩阵,每行对应一个多径分量。
        max_doppler_hz > 0 时每条径独立使用 Jakes 模型产生时变系数。
        """
        pdp = self.cfg.pdp
        n_taps = len(pdp)
        taps = np.zeros((n_taps, n), dtype=complex)

        for i, (_, power_db) in enumerate(pdp):
            amp = 10.0 ** (power_db / 20.0)
            if self.cfg.max_doppler_hz <= 0:
                h = (self._rng.standard_normal() + 1j * self._rng.standard_normal()) / np.sqrt(2)
                taps[i, :] = amp * h
            else:
                taps[i, :] = amp * self._jakes_fading(n)

        # 归一化: 使平均总功率为 1
        total_power = np.sum([10.0 ** (p / 10.0) for _, p in pdp])
        taps /= np.sqrt(total_power)

        return taps

    # ── AWGN ──

    def _add_noise(self, signal: np.ndarray, snr_db: float, sps: int = 1) -> np.ndarray:
        snr_linear = 10.0 ** (snr_db / 10.0)
        signal_power = np.mean(np.abs(signal) ** 2) * sps
        if signal_power < 1e-20:
            return signal.copy()
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            self._rng.standard_normal(signal.shape)
            + 1j * self._rng.standard_normal(signal.shape)
        )
        return signal + noise

    # ── Doppler 统计验证 ──

    def doppler_autocorrelation(self, n: int, max_lag: int = 64) -> np.ndarray:
        """计算 Jakes 衰落序列的归一化自相关函数, 用于验证 J₀ 特性。

        返回长度为 max_lag 的实数数组, r[k] = Re{E[h(n)·h*(n+k)]} / E[|h|²]。
        """
        h = self._jakes_fading(n + max_lag)
        r = np.zeros(max_lag)
        power = np.mean(np.abs(h) ** 2)
        if power < 1e-20:
            return r
        for k in range(max_lag):
            r[k] = np.real(np.mean(h[:n] * np.conj(h[k:k + n]))) / power
        return r

    def theoretical_autocorrelation(self, max_lag: int = 64) -> np.ndarray:
        """理论自相关函数 J₀(2π·f_d·k/f_s), 供对比验证。"""
        fd = self.cfg.max_doppler_hz
        fs = self.cfg.symbol_rate_hz
        k = np.arange(max_lag)
        return bessel_j0(2 * np.pi * fd * k / fs)


# ── 多用户干扰模型 (TXS-10002-2025 §8.3.2) ──


@dataclass
class InterferenceConfig:
    """干扰源配置。

    :ivar sir_db: 信号与干扰功率比 (dB), 即 S/I。
    :ivar freq_offset_hz: 干扰信号中心频率相对有用信号的偏移 (Hz)。
        0 表示同信道干扰。
    :ivar interferer_bw_hz: 干扰信号带宽 (Hz), 用于频谱成型。
        0 表示与有用信号相同带宽 (不做滤波)。
    :ivar seed: 随机种子。
    """
    sir_db: float = 10.0
    freq_offset_hz: float = 0.0
    interferer_bw_hz: float = 0.0
    seed: int | None = None


def add_interference(
    signal: np.ndarray,
    interferers: list[InterferenceConfig],
    sample_rate_hz: float = 1e6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """向信号叠加多个干扰源。

    每个干扰源独立生成与有用信号等长的随机 IQ 序列, 按 SIR 缩放后叠加。
    支持同信道和邻信道干扰 (通过频率偏移实现)。

    :param signal: 有用信号 (复数 IQ)。
    :param interferers: 干扰源配置列表。
    :param sample_rate_hz: 采样率 (Hz)。
    :param rng: 可选随机数生成器。

    :returns: 叠加干扰后的信号。
    """
    n = len(signal)
    result = signal.copy()
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power < 1e-20:
        return result

    for intf in interferers:
        local_rng = np.random.default_rng(intf.seed) if intf.seed is not None else (
            rng if rng is not None else np.random.default_rng()
        )

        # 生成干扰基带信号 (随机 QPSK 类似)
        intf_signal = (
            local_rng.standard_normal(n) + 1j * local_rng.standard_normal(n)
        ) / np.sqrt(2)

        # 按 SIR 缩放: P_intf = P_sig / SIR_linear
        sir_lin = 10.0 ** (intf.sir_db / 10.0)
        intf_power = sig_power / sir_lin
        intf_signal *= np.sqrt(intf_power / np.mean(np.abs(intf_signal) ** 2))

        # 频率偏移 (邻信道干扰)
        if intf.freq_offset_hz != 0.0:
            t = np.arange(n) / sample_rate_hz
            intf_signal *= np.exp(1j * 2 * np.pi * intf.freq_offset_hz * t)

        result += intf_signal

    return result


def compute_sinr(
    signal: np.ndarray,
    noise_signal: np.ndarray,
    interference_signal: np.ndarray,
) -> float:
    """计算 SINR (dB)。

    :param signal: 纯净有用信号。
    :param noise_signal: 含噪声的接收信号 (信号+噪声, 不含干扰)。
    :param interference_signal: 干扰叠加后的接收信号 (信号+噪声+干扰)。

    :returns: SINR (dB)。
    """
    s_power = np.mean(np.abs(signal) ** 2)
    n_power = np.mean(np.abs(noise_signal - signal) ** 2)
    i_power = np.mean(np.abs(interference_signal - noise_signal) ** 2)
    denominator = n_power + i_power
    if denominator < 1e-20:
        return 100.0
    return float(10.0 * np.log10(s_power / denominator))
