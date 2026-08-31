"""TXS-10002-2025 6.10.3 跳频序列与频率管理。

实现标准定义的跳频机制, 包括:
- 射频信道/物理信道映射与频率表管理 (8.1.2)
- 伪随机数生成器 (6.10.3.2)
- 数据链路跳频 (6.10.3.3)
- 系统管理帧链路跳频 (6.10.3.4)
- 测量链路跳频 (6.10.3.5)
"""

from __future__ import annotations

__all__ = [
    "BAND_2400",
    "BAND_5100",
    "BAND_5800",
    "MEAS_HOP_ASCENDING",
    "MEAS_HOP_DESCENDING",
    "MEAS_HOP_RANDOM",
    "FreqTable",
    "MeasLinkHopper",
    "channel_to_freq",
    "data_link_hop",
    "derive_hop_param2",
    "freq_to_channel",
    "generate_hopping_sequence",
    "hopping_prng",
    "mgmt_frame_hop",
]


from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 8.1.2 射频信道中心频率与射频信道号
# ---------------------------------------------------------------------------

# 2.4 GHz 频段: Fc = 2402 + N MHz, N = 0..78, NOffset = 0
# 广播信道: 物理信道号 76, 77, 78 (射频信道 0, 22, 78; 频率 2402, 2424, 2480 MHz)
# 数据信道: 物理信道号 0..75 (76 个信道)

BAND_2400 = "2400"
BAND_5100 = "5100"
BAND_5800 = "5800"

_BAND_PARAMS: dict[str, dict] = {
    BAND_2400: {
        "f_band_low": 2402,
        "n_offset": 0,
        "n_min": 0,
        "n_max": 78,
        "broadcast_phys": [76, 77, 78],
        "data_mod": 76,          # 初次频点映射取模值
        "data_offset": 0,        # 映射偏移
    },
    BAND_5100: {
        "f_band_low": 5152,
        "n_offset": 79,
        "n_min": 79,
        "n_max": 275,
        "broadcast_phys": [273, 274, 275],
        "data_mod": 194,
        "data_offset": 79,
    },
    BAND_5800: {
        "f_band_low": 5727,
        "n_offset": 276,
        "n_min": 276,
        "n_max": 397,
        "broadcast_phys": [],
        "data_mod": 119,
        "data_offset": 276,
    },
}

# 2.4 GHz 2 MHz 带宽可用物理信道号 (射频信道号为偶数且非广播, 表 45)
_2M_PHYS_2400 = [
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
    22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
    62, 64, 66, 68, 70, 72, 74,
]

# 2.4 GHz 4 MHz 带宽可用物理信道号 (表 45)
_4M_PHYS_2400 = [
    2, 6, 10, 14, 18, 25, 29, 33, 37, 41,
    45, 49, 53, 57, 61, 65, 69, 73,
]


def channel_to_freq(channel_num: int, band: str = BAND_2400) -> float:
    """射频信道号转中心频率 (MHz)。"""
    p = _BAND_PARAMS[band]
    return p["f_band_low"] + (channel_num - p["n_offset"])


def freq_to_channel(freq_mhz: float, band: str = BAND_2400) -> int:
    """中心频率 (MHz) 转射频信道号。"""
    p = _BAND_PARAMS[band]
    return round(freq_mhz - p["f_band_low"]) + p["n_offset"]


# ---------------------------------------------------------------------------
# 频点表管理
# ---------------------------------------------------------------------------

@dataclass
class FreqTable:
    """信道频率表管理。

    管理完整频点表和可用频点表, 支持 1M/2M/4M 带宽场景。

    :ivar band: 频段标识, "2400" / "5100" / "5800"。
    :ivar bandwidth_mhz: 信道带宽, 1 / 2 / 4 MHz。
    :ivar blocked_channels: 被阻塞(不可用)的物理信道号集合。
    """
    band: str = BAND_2400
    bandwidth_mhz: int = 1
    blocked_channels: set[int] = field(default_factory=set)

    def full_table(self) -> list[int]:
        """完整频点表: 对应频段除去广播信道的所有频点, 按物理信道号升序。"""
        p = _BAND_PARAMS[self.band]
        broadcast = set(p["broadcast_phys"])
        all_ch = range(p["n_min"], p["n_max"] + 1)
        full = sorted(ch for ch in all_ch if ch not in broadcast)
        return self._filter_bandwidth(full)

    def available_table(self) -> list[int]:
        """可用频点表: 完整表中去除阻塞信道, 按物理信道号升序。"""
        return [ch for ch in self.full_table() if ch not in self.blocked_channels]

    def _filter_bandwidth(self, channels: list[int]) -> list[int]:
        """根据带宽过滤可用信道。"""
        if self.bandwidth_mhz == 1:
            return channels
        if self.band == BAND_2400:
            if self.bandwidth_mhz == 2:
                valid = set(_2M_PHYS_2400)
            elif self.bandwidth_mhz == 4:
                valid = set(_4M_PHYS_2400)
            else:
                return channels
            return [ch for ch in channels if ch in valid]
        return channels


# ---------------------------------------------------------------------------
# 6.10.3.2 伪随机数生成器
# ---------------------------------------------------------------------------

def _bit_reverse_byte(b: int) -> int:
    """将一个 8 位整数的比特反转。例如 0b10000000 -> 0b00000001。"""
    result = 0
    for _ in range(8):
        result = (result << 1) | (b & 1)
        b >>= 1
    return result


def _reverse_16(x: int) -> int:
    """对 16 位整数的低 8 位和高 8 位分别进行位反转。"""
    low8 = x & 0xFF
    high8 = (x >> 8) & 0xFF
    return (_bit_reverse_byte(high8) << 8) | _bit_reverse_byte(low8)


def hopping_prng(hop_param1: int, hop_param2: int) -> int:
    """标准 6.10.3.2 伪随机数生成器。

    :param hop_param1: 16 位跳频参数 1, 通常为调度时隙计数的低 16 位。
    :param hop_param2: 16 位跳频参数 2, 由同步序列/逻辑链路标识/随机种子派生。

    :returns: 16 位伪随机数。
    """
    hop_param1 &= 0xFFFF
    hop_param2 &= 0xFFFF
    x = hop_param1 ^ hop_param2
    for _ in range(3):
        x = _reverse_16(x)
        x = (x * 17 + hop_param2) & 0xFFFF  # mod 2^16
    x = x ^ hop_param2
    return x


def derive_hop_param2(identifier: int, bit_width: int = 32) -> int:
    """从同步序列或逻辑链路标识派生跳频参数 2。

    标准 6.10.3.2:
    - 64 位同步序列: 截取低 32 位
    - 32 位同步序列: 直接使用
    - 24 位逻辑链路标识: 高位补零至 32 位
    低 16 和高 16 位异或后作为 hop_param2。

    :param identifier: 同步序列值或逻辑链路标识。
    :param bit_width: 标识符位宽 (24/32/64)。

    :returns: 16 位 hop_param2。
    """
    if bit_width == 64:
        val = identifier & 0xFFFFFFFF  # 取低 32 位
    elif bit_width == 24:
        val = identifier & 0xFFFFFF  # 24 位, 高位补零
    else:
        val = identifier & 0xFFFFFFFF
    low16 = val & 0xFFFF
    high16 = (val >> 16) & 0xFFFF
    return low16 ^ high16


# ---------------------------------------------------------------------------
# 6.10.3.3 数据链路跳频
# ---------------------------------------------------------------------------

def data_link_hop(slot_counter: int, hop_param2: int,
                  freq_table: FreqTable) -> int:
    """数据链路跳频: 根据时隙计数和跳频参数确定跳频频点。

    标准 6.10.3.3: PRNG → 初次频点映射 → (无效时) 可用频点映射。

    :param slot_counter: 当前调度时隙计数。
    :param hop_param2: 16 位跳频参数 2。
    :param freq_table: 频率表管理对象。

    :returns: 选中的物理信道号。
    """
    hop_param1 = slot_counter & 0xFFFF
    rand16 = hopping_prng(hop_param1, hop_param2)

    # 初次频点映射 (6.10.3.3.1)
    p = _BAND_PARAMS[freq_table.band]
    phys_ch = (rand16 % p["data_mod"]) + p["data_offset"]

    # 检查频点有效性
    full = freq_table.full_table()
    avail = freq_table.available_table()

    if freq_table.bandwidth_mhz == 1:
        if phys_ch in avail:
            return phys_ch
    else:
        # 2M/4M: 检查是否属于对应带宽的有效频点
        bw_channels = set(full)
        if phys_ch in bw_channels and phys_ch in avail:
            return phys_ch

    # 可用频点映射 (6.10.3.3.2)
    return _available_freq_map(rand16, avail)


def _available_freq_map(rand16: int, available: list[int]) -> int:
    """可用频点映射过程。

    标准 6.10.3.3.2 / 6.10.3.4.1 / 6.10.3.5.1:
    index = floor(rand16 * N_available / 2^16), 然后映射到可用频点表。

    :param rand16: 16 位伪随机数。
    :param available: 可用频点表 (升序排列)。

    :returns: 选中的物理信道号。

    :raises ValueError: 可用频点表为空。
    """
    if not available:
        raise ValueError("可用频点表为空")
    idx = (rand16 * len(available)) >> 16  # floor(rand16 * N / 2^16)
    idx = min(idx, len(available) - 1)
    return available[idx]


# ---------------------------------------------------------------------------
# 6.10.3.4 系统管理帧链路跳频
# ---------------------------------------------------------------------------

def mgmt_frame_hop(slot_counter: int, hop_param2: int,
                   freq_table: FreqTable) -> int:
    """系统管理帧链路跳频。

    标准 6.10.3.4: PRNG → 可用频点映射, 直接在可用频点表上映射。

    :param slot_counter: 当前调度时隙计数。
    :param hop_param2: 16 位跳频参数 2。
    :param freq_table: 频率表管理对象。

    :returns: 选中的物理信道号。
    """
    hop_param1 = slot_counter & 0xFFFF
    rand16 = hopping_prng(hop_param1, hop_param2)
    return _available_freq_map(rand16, freq_table.available_table())


# ---------------------------------------------------------------------------
# 6.10.3.5 测量链路跳频
# ---------------------------------------------------------------------------

MEAS_HOP_ASCENDING = 0   # 方案 1: 射频信道号从低到高
MEAS_HOP_DESCENDING = 1  # 方案 2: 射频信道号从高到低
MEAS_HOP_RANDOM = 2      # 方案 3: 随机跳频


@dataclass
class MeasLinkHopper:
    """测量链路跳频器。

    标准 6.10.3.5: 支持三种跳频方案。
    方案 3 在使用后从可用频点表中删除已用频点, 表空时重置。

    :ivar mode: 跳频模式, 0/1/2。
    :ivar freq_table: 频率表对象。
    :ivar hop_param2: 跳频参数 2 (方案 3 使用)。
    :ivar init_channel: 初始化阶段频点 (可选)。
    """
    mode: int = MEAS_HOP_ASCENDING
    freq_table: FreqTable = field(default_factory=FreqTable)
    hop_param2: int = 0
    init_channel: int | None = None

    def __post_init__(self) -> None:
        self._current_available: list[int] = list(self.freq_table.available_table())
        self._seq_index: int = 0
        self._initialized: bool = False

    def reset(self) -> None:
        """重置可用频点表为初始状态。"""
        self._current_available = list(self.freq_table.available_table())
        self._seq_index = 0
        self._initialized = False

    def next_channel(self, slot_counter: int = 0,
                     is_init_phase: bool = False) -> int:
        """获取下一个跳频信道。

        :param slot_counter: 当前调度时隙计数 (方案 3 使用)。
        :param is_init_phase: 是否处于初始化阶段。

        :returns: 物理信道号。
        """
        if is_init_phase and self.init_channel is not None:
            return self.init_channel

        avail = self.freq_table.available_table()

        if self.mode == MEAS_HOP_ASCENDING:
            ch = avail[self._seq_index % len(avail)]
            self._seq_index += 1
            return ch

        if self.mode == MEAS_HOP_DESCENDING:
            rev = list(reversed(avail))
            ch = rev[self._seq_index % len(rev)]
            self._seq_index += 1
            return ch

        # 方案 3: 随机跳频
        if not self._current_available:
            self._current_available = list(avail)

        hop_param1 = slot_counter & 0xFFFF
        rand16 = hopping_prng(hop_param1, self.hop_param2)
        idx = (rand16 * len(self._current_available)) >> 16
        idx = min(idx, len(self._current_available) - 1)
        ch = self._current_available[idx]

        if not is_init_phase:
            self._current_available.pop(idx)

        return ch


# ---------------------------------------------------------------------------
# 跳频序列生成 (便捷函数)
# ---------------------------------------------------------------------------

def generate_hopping_sequence(n_hops: int, hop_param2: int,
                              freq_table: FreqTable | None = None,
                              link_type: str = "data",
                              start_slot: int = 0) -> list[int]:
    """生成指定长度的跳频序列。

    :param n_hops: 需要的跳频次数。
    :param hop_param2: 跳频参数 2。
    :param freq_table: 频率表对象, 默认 2.4 GHz 1 MHz 全可用。
    :param link_type: "data" / "mgmt" / "meas_asc" / "meas_desc" / "meas_rand"。
    :param start_slot: 起始时隙号。

    :returns: 物理信道号列表。
    """
    if freq_table is None:
        freq_table = FreqTable()

    if link_type == "data":
        return [data_link_hop(start_slot + i, hop_param2, freq_table)
                for i in range(n_hops)]

    if link_type == "mgmt":
        return [mgmt_frame_hop(start_slot + i, hop_param2, freq_table)
                for i in range(n_hops)]

    # 测量链路
    mode_map = {
        "meas_asc": MEAS_HOP_ASCENDING,
        "meas_desc": MEAS_HOP_DESCENDING,
        "meas_rand": MEAS_HOP_RANDOM,
    }
    hopper = MeasLinkHopper(
        mode=mode_map.get(link_type, MEAS_HOP_ASCENDING),
        freq_table=freq_table,
        hop_param2=hop_param2,
    )
    return [hopper.next_channel(start_slot + i) for i in range(n_hops)]
