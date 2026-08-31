"""导频符号插入与提取 -- TXS-10002-2025 标准 6.2.1.2 节。"""


__all__ = [
    "EVEN_ROTATION_DEG",
    "PILOT_PHASE_DEG",
    "insert_pilots",
    "pilot_symbol",
    "remove_pilots",
]


import numpy as np

# 各调制方式的导频参考相位（单位：度）
# 标准定义: BPSK导频90°, QPSK导频45°, 8PSK导频22.5°
PILOT_PHASE_DEG = {
    "BPSK": 90.0,
    "QPSK": 45.0,
    "8PSK": 22.5,
}

# 各调制方式偶数符号的相位旋转量（顺时针，单位：度）
EVEN_ROTATION_DEG = {
    "BPSK": -90.0,
    "QPSK": -45.0,
    "8PSK": -22.5,
}


def pilot_symbol(mod_type: str, symbol_index: int) -> complex:
    """生成带适当相位旋转的单个导频符号。

    :param mod_type: 调制方式，取 "BPSK"、"QPSK" 或 "8PSK"。
    :param symbol_index: 帧内以 0 为基的符号索引（决定奇偶位置）。
                 标准规定第一个符号处于奇数位置。

    :returns: 复数导频符号。
    """
    phase_deg = PILOT_PHASE_DEG[mod_type]
    # "无线帧的第一个符号认为处于奇数位" → index 0 is odd, index 1 is even, ...
    is_even = (symbol_index % 2) == 1
    if is_even:
        phase_deg += EVEN_ROTATION_DEG[mod_type]
    return np.exp(1j * np.deg2rad(phase_deg))


def insert_pilots(
    data_symbols: np.ndarray,
    pilot_interval: int,
    mod_type: str,
    start_symbol_index: int = 0,
    omit_last_pilot: bool = True,
) -> tuple[np.ndarray, int]:
    """向数据符号流中插入导频符号。

    每隔 `pilot_interval` 个数据符号后插入一个导频符号。
    按标准规定，帧中最后一个导频应省略。

    :param data_symbols: 复数数据符号数组。
    :param pilot_interval: 标准中的 N 值（4、8 或 16）。
    :param mod_type: 导频相位所用的调制方式。
    :param start_symbol_index: 帧内第一个数据符号的绝对符号索引。
    :param omit_last_pilot: 是否省略最后一个导频符号。

    :returns: (output_symbols, total_symbol_count) — 插入导频后的符号数组及总符号数。
    """
    if pilot_interval <= 0:
        return data_symbols.copy(), len(data_symbols)

    result = []
    sym_idx = start_symbol_index
    n_data = len(data_symbols)
    data_pos = 0
    pilot_positions = []

    while data_pos < n_data:
        # 取最多 pilot_interval 个数据符号
        chunk_end = min(data_pos + pilot_interval, n_data)
        chunk = data_symbols[data_pos:chunk_end]
        result.append(chunk)
        sym_idx += len(chunk)
        data_pos = chunk_end

        # 若当前块已满则在其后插入导频
        if len(chunk) == pilot_interval and data_pos <= n_data:
            pilot_positions.append(len(result))
            p = pilot_symbol(mod_type, sym_idx)
            result.append(np.array([p]))
            sym_idx += 1

    output = np.concatenate(result) if result else np.array([], dtype=complex)

    # 按需省略最后一个导频
    if omit_last_pilot and pilot_positions and len(output) > 0:
        # 检查最后一个元素是否为导频
        # 最后一个导频位置：找到最后插入的导频
        # 检查输出的最后一个符号是否为导频
        total_data_inserted = n_data
        n_pilots = len(pilot_positions)
        if n_pilots > 0 and total_data_inserted % pilot_interval == 0:
            # 最后一个符号是导频 → 移除
            output = output[:-1]

    return output, len(output)


def remove_pilots(
    symbols: np.ndarray,
    pilot_interval: int,
    last_pilot_omitted: bool = True,
) -> np.ndarray:
    """从接收符号流中移除导频符号。

    :param symbols: 含导频的接收符号。
    :param pilot_interval: N 值（4、8 或 16）。
    :param last_pilot_omitted: 发送侧是否已省略最后一个导频。

    :returns: 移除导频后的数据符号。
    """
    if pilot_interval <= 0:
        return symbols.copy()

    stride = pilot_interval + 1  # data + pilot
    result = []
    pos = 0

    while pos < len(symbols):
        remaining = len(symbols) - pos
        if remaining >= stride:
            # 完整组：取数据，跳过导频
            result.append(symbols[pos : pos + pilot_interval])
            pos += stride
        else:
            # 末尾不完整组（无导频或导频已省略）
            result.append(symbols[pos:])
            pos = len(symbols)

    return np.concatenate(result) if result else np.array([], dtype=complex)
