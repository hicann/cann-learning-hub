"""TXS-10002-2025 均衡器: ZF / MMSE 频域与时域均衡。

对于 SparkLink SLE 的 PSK 调制 (帧类型 2-4),接收端需要
信道均衡来对抗多径衰落引起的符号间干扰 (ISI)。
"""


__all__ = [
    "equalize_1tap",
    "equalize_mmse_freq",
    "equalize_mmse_time",
    "equalize_zf",
    "estimate_channel_freq",
]


import numpy as np


def equalize_zf(rx_signal: np.ndarray, channel_freq: np.ndarray) -> np.ndarray:
    """零强制 (ZF) 频域均衡。

    :param rx_signal: 接收信号 (时域)。
    :param channel_freq: 信道频率响应 H[k], 长度等于 rx_signal 的 FFT 长度。

    :returns: 均衡后的时域信号。
    """
    n = len(rx_signal)
    rx_freq = np.fft.fft(rx_signal, n)
    # 避免除零
    h_safe = np.where(np.abs(channel_freq[:n]) > 1e-10, channel_freq[:n], 1e-10)
    eq_freq = rx_freq / h_safe
    return np.fft.ifft(eq_freq, n)


def equalize_mmse_freq(
    rx_signal: np.ndarray,
    channel_freq: np.ndarray,
    noise_var: float,
) -> np.ndarray:
    """MMSE 频域均衡。

    H_mmse[k] = conj(H[k]) / (|H[k]|^2 + sigma^2)

    :param rx_signal: 接收信号 (时域)。
    :param channel_freq: 信道频率响应 H[k]。
    :param noise_var: 噪声方差 sigma^2。

    :returns: 均衡后的时域信号。
    """
    n = len(rx_signal)
    rx_freq = np.fft.fft(rx_signal, n)
    h = channel_freq[:n]
    w = np.conj(h) / (np.abs(h) ** 2 + noise_var)
    eq_freq = rx_freq * w
    return np.fft.ifft(eq_freq, n)


def estimate_channel_freq(
    rx_signal: np.ndarray,
    tx_known: np.ndarray,
    n_fft: int | None = None,
) -> np.ndarray:
    """基于已知训练序列的频域信道估计 (LS)。

    H_est[k] = Y[k] / X[k]

    :param rx_signal: 接收到的训练序列。
    :param tx_known: 已知的发送训练序列。
    :param n_fft: FFT 长度, 默认为信号长度。

    :returns: 估计的信道频率响应。
    """
    if n_fft is None:
        n_fft = max(len(rx_signal), len(tx_known))

    rx_freq = np.fft.fft(rx_signal, n_fft)
    tx_freq = np.fft.fft(tx_known, n_fft)

    # LS 估计, 避免除零
    tx_safe = np.where(np.abs(tx_freq) > 1e-10, tx_freq, 1e-10)
    h_est = rx_freq / tx_safe

    return h_est


def equalize_mmse_time(
    rx_signal: np.ndarray,
    channel_taps: np.ndarray,
    noise_var: float,
    n_taps_eq: int = 11,
) -> np.ndarray:
    """MMSE 时域均衡器 (FIR Wiener 滤波器)。

    计算 MMSE 最优 FIR 滤波器系数:
        w = R^{-1} p
    其中 R = h自相关 + 噪声, p = h与期望延迟的互相关。

    :param rx_signal: 接收信号。
    :param channel_taps: 信道冲激响应 h[0], h[1], ..., h[L-1]。
    :param noise_var: 噪声方差。
    :param n_taps_eq: 均衡器 FIR 长度。

    :returns: 均衡后的信号。
    """
    h = np.asarray(channel_taps, dtype=complex).flatten()
    l_h = len(h)
    l_eq = n_taps_eq

    # 自相关: r_hh[k] = sum_l h[l] * conj(h[l-k])
    r_hh = np.correlate(h, h, mode='full')
    center_rh = l_h - 1

    # 构造 Toeplitz R 矩阵
    R = np.zeros((l_eq, l_eq), dtype=complex)
    for i in range(l_eq):
        for j in range(l_eq):
            lag = i - j
            idx = center_rh + lag
            if 0 <= idx < len(r_hh):
                R[i, j] = r_hh[idx]
    R += noise_var * np.eye(l_eq)

    # 最优延迟 (mode='same' 引入的组延迟补偿)
    delay = (l_eq - 1) // 2

    # 互相关向量 p[i] = conj(h[delay - i])
    p = np.zeros(l_eq, dtype=complex)
    for i in range(l_eq):
        k = delay - i
        if 0 <= k < l_h:
            p[i] = np.conj(h[k])

    # 求解 Wiener-Hopf 方程
    w = np.linalg.solve(R, p)

    # 应用 FIR 滤波
    return np.convolve(rx_signal, w, mode='same')


def equalize_1tap(
    rx_symbols: np.ndarray,
    h_coeffs: np.ndarray,
    noise_var: float = 0.0,
    method: str = "mmse",
) -> np.ndarray:
    """逐符号 1-tap 均衡 (平坦衰落信道)。

    :param rx_symbols: 接收符号。
    :param h_coeffs: 每个符号的信道系数 (与 rx_symbols 等长)。
    :param noise_var: 噪声方差, 仅 MMSE 使用。
    :param method: "zf" 或 "mmse"。

    :returns: 均衡后的符号。
    """
    h = np.asarray(h_coeffs, dtype=complex)
    if method == "zf":
        h_safe = np.where(np.abs(h) > 1e-10, h, 1e-10 + 0j)
        return rx_symbols / h_safe
    elif method == "mmse":
        w = np.conj(h) / (np.abs(h) ** 2 + noise_var)
        return rx_symbols * w
    else:
        raise ValueError(f"Unknown method: {method}")
