"""TXS-10002-2025 链路仿真: GFSK/PSK 无编码 + Polar编码端到端仿真。"""


__all__ = [
    "run_phase1_simulation",
    "run_phase2_simulation",
    "run_phase3_simulation",
    "run_phase4_simulation",
    "run_phase5_simulation",
    "run_phase6_simulation",
    "run_phase7_simulation",
    "run_phase8_simulation",
    "run_phase9_simulation",
    "run_phase10_simulation",
    "run_phase11_simulation",
    "run_phase12_simulation",
    "run_phase13_simulation",
    "run_phase14_simulation",
    "run_phase15_simulation",
    "run_phase16_simulation",
    "sim_access_scheduled_link",
    "sim_amc_throughput",
    "sim_channel_eq_link",
    "sim_doppler_link",
    "sim_doppler_multipath_link",
    "sim_dual_node_link",
    "sim_dual_node_mcs_adapt",
    "sim_dual_node_secure_link",
    "sim_encrypted_vs_plain",
    "sim_event_group_timing",
    "sim_frame_link",
    "sim_gfsk_link",
    "sim_harq_link",
    "sim_hopping_link",
    "sim_hopping_multipath_link",
    "sim_mac_data_link",
    "sim_mac_mux_link",
    "sim_mac_signaling_link",
    "sim_multi_link",
    "sim_multi_user_interference",
    "sim_node_access_flow",
    "sim_node_channel_sweep",
    "sim_node_hopping_link",
    "sim_node_measurement",
    "sim_node_power_adapt",
    "sim_pairing_signaling_phy",
    "sim_pipeline_channel_link",
    "sim_pipeline_link",
    "sim_polar_coded_psk_link",
    "sim_psk_link",
    "sim_qos_amc_adaptive",
    "sim_qos_arq_link",
    "sim_qos_flow_control",
    "sim_secure_link",
    "sim_sir_sweep",
    "sim_superframe_capacity",
]


import os
from pathlib import Path

import numpy as np

from nearlink_sdr.common.polar import PolarEncoder, get_info_bit_count, get_polar_decoder
from nearlink_sdr.phy.channel import ChannelModel
from nearlink_sdr.phy.gfsk import GFSKDemodulator, GFSKModulator
from nearlink_sdr.phy.preamble import generate_preamble
from nearlink_sdr.phy.psk import PSKDemodulator, PSKModulator
from nearlink_sdr.phy.sync_sequence import (
    sync_signal_1,
)

# 仿真输出目录: 可通过环境变量 NEARLINK_SDR_OUTPUT 覆盖
_OUTPUT_DIR = Path(os.environ.get("NEARLINK_SDR_OUTPUT", "output"))


def _ensure_output_dir() -> Path:
    """确保输出目录存在并返回路径。"""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def _ber(tx: np.ndarray, rx: np.ndarray) -> float:
    n = min(len(tx), len(rx))
    if n == 0:
        return 0.0
    return float(np.sum(tx[:n] != rx[:n])) / n


def sim_gfsk_link(num_data_bits: int = 1000,
                  snr_range_db: np.ndarray | None = None,
                  sps: int = 8,
                  seed: int = 42) -> dict:
    """帧类型1 GFSK无编码链路仿真。

    流程: 前导码 + 同步信号1(广播) + 数据 → GFSK调制 → AWGN信道 → GFSK解调 → BER

    :returns: {"snr_db": [...], "ber": [...]}
    """
    if snr_range_db is None:
        snr_range_db = np.arange(0, 18, 2)

    rng = np.random.default_rng(seed)
    mod = GFSKModulator(sps=sps)
    demod = GFSKDemodulator(sps=sps)

    # 构造帧比特: 前导 + 同步序列 + 数据
    preamble = generate_preamble(1, symbol_rate_mhz=1.0)
    sync = sync_signal_1(None)
    data_bits = rng.integers(0, 2, num_data_bits)
    frame_bits = np.concatenate([preamble, sync, data_bits])

    n_overhead = len(preamble) + len(sync)
    ber_list = []

    for snr in snr_range_db:
        ch = ChannelModel(snr_db=float(snr))
        tx_signal = mod.modulate(frame_bits)
        rx_signal = ch.apply_awgn(tx_signal, sps)
        rx_bits = demod.demodulate(rx_signal)
        # 提取数据段
        rx_data = rx_bits[n_overhead:n_overhead + num_data_bits]
        ber_list.append(_ber(data_bits, rx_data))

    return {"snr_db": snr_range_db.tolist(), "ber": ber_list}


def sim_psk_link(num_data_bits: int = 1000,
                 mod_type: str = "QPSK",
                 frame_type: int = 2,
                 snr_range_db: np.ndarray | None = None,
                 sps: int = 4,
                 seed: int = 42) -> dict:
    """帧类型2/3/4 PSK无编码链路仿真。

    流程: 数据 → PSK调制 → AWGN信道 → PSK解调 → BER
    (前导和同步信号作为独立信号段，不影响数据BER)

    :returns: {"snr_db": [...], "ber": [...]}
    """
    if snr_range_db is None:
        snr_range_db = np.arange(0, 18, 2)

    rng = np.random.default_rng(seed)
    mod = PSKModulator(mod_type=mod_type, sps=sps)
    demod = PSKDemodulator(mod_type=mod_type, sps=sps)

    data_bits = rng.integers(0, 2, num_data_bits)
    ber_list = []

    for snr in snr_range_db:
        tx_signal = mod.modulate(data_bits)              # 符号映射 + RRC 成型
        ch = ChannelModel(snr_db=float(snr))
        rx_signal = ch.apply_awgn(tx_signal, sps)        # 在成型后加噪, sps 补偿功率
        rx_bits = demod.demodulate(rx_signal)
        ber_list.append(_ber(data_bits, rx_bits))

    return {"snr_db": snr_range_db.tolist(), "ber": ber_list}


def run_phase1_simulation():
    """Phase 1 综合仿真: GFSK + BPSK + QPSK BER曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(0, 18, 2)

    print("=== Phase 1 Link Simulation ===")
    print()

    # GFSK (帧类型1)
    print("[1/3] GFSK link simulation...")
    gfsk_result = sim_gfsk_link(num_data_bits=5000, snr_range_db=snr_range)
    for s, b in zip(gfsk_result["snr_db"], gfsk_result["ber"], strict=False):
        print(f"  SNR={s:2d} dB  BER={b:.5f}")

    # BPSK (帧类型3/4)
    print("[2/3] BPSK link simulation...")
    bpsk_result = sim_psk_link(num_data_bits=5000, mod_type="BPSK",
                               frame_type=3, snr_range_db=snr_range)
    for s, b in zip(bpsk_result["snr_db"], bpsk_result["ber"], strict=False):
        print(f"  SNR={s:2d} dB  BER={b:.5f}")

    # QPSK (帧类型2)
    print("[3/3] QPSK link simulation...")
    qpsk_result = sim_psk_link(num_data_bits=5000, mod_type="QPSK",
                               frame_type=2, snr_range_db=snr_range)
    for s, b in zip(qpsk_result["snr_db"], qpsk_result["ber"], strict=False):
        print(f"  SNR={s:2d} dB  BER={b:.5f}")

    # BER曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(gfsk_result["snr_db"], gfsk_result["ber"],
                "o-", label="GFSK (Frame Type 1)")
    ax.semilogy(bpsk_result["snr_db"], bpsk_result["ber"],
                "s-", label="BPSK (Frame Type 3/4)")
    ax.semilogy(qpsk_result["snr_db"], qpsk_result["ber"],
                "^-", label="QPSK (Frame Type 2)")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("SparkLink SLE PHY - Phase 1 Uncoded BER")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim(bottom=1e-5)
    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase1.png", dpi=150)
    print("\nBER curve saved to ber_phase1.png")


# ── Phase 2: Polar编码链路仿真 ──


def sim_polar_coded_psk_link(
    num_info_bits: int = 1000,
    mod_type: str = "BPSK",
    rate_str: str = "1/2",
    code_length: int = 256,
    snr_range_db: np.ndarray | None = None,
    sps: int = 4,
    seed: int = 42,
) -> dict:
    """Polar编码PSK链路仿真。

    流程: 信息比特 → Polar编码 → BPSK/QPSK调制 → AWGN信道 → 解调(软判决LLR) → SC解码 → BER

    :returns: {"snr_db": [...], "ber": [...], "fer": [...]}
    """
    if snr_range_db is None:
        snr_range_db = np.arange(-2, 12, 1)

    K = get_info_bit_count(rate_str, code_length)
    enc = PolarEncoder(code_length, K)
    dec = get_polar_decoder(code_length, K)

    rng = np.random.default_rng(seed)

    # 将总信息比特分成多个码块
    n_blocks = max(1, num_info_bits // K)

    ber_list = []
    fer_list = []

    for snr in snr_range_db:
        total_bit_errors = 0
        total_bits = 0
        frame_errors = 0

        for _ in range(n_blocks):
            info = rng.integers(0, 2, size=K, dtype=np.int8)
            coded = enc.encode(info)

            # BPSK映射: 0 → +1, 1 → -1
            bpsk = 1.0 - 2.0 * coded.astype(np.float64)

            # AWGN信道
            snr_linear = 10.0 ** (float(snr) / 10.0)
            # 对于码率R的编码, Eb/N0 = SNR / R
            R_val = K / code_length
            noise_var = 1.0 / (2.0 * R_val * snr_linear) if snr_linear > 0 else 1e10
            noise = rng.normal(0, np.sqrt(noise_var), size=code_length)
            received = bpsk + noise

            # LLR计算: LLR = 2*y/sigma^2
            llr = 2.0 * received / noise_var

            # SC解码
            decoded = dec.decode(llr)

            # 统计
            bit_errors = int(np.sum(decoded != info))
            total_bit_errors += bit_errors
            total_bits += K
            if bit_errors > 0:
                frame_errors += 1

        ber = total_bit_errors / total_bits if total_bits > 0 else 0.0
        fer = frame_errors / n_blocks
        ber_list.append(ber)
        fer_list.append(fer)

    return {
        "snr_db": snr_range_db.tolist(),
        "ber": ber_list,
        "fer": fer_list,
        "code_params": f"N={code_length}, K={K}, R={rate_str}",
    }


def run_phase2_simulation():
    """Phase 2 综合仿真: Polar编码 BPSK/QPSK BER/FER 曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(-2, 10, 0.5)

    print("=== Phase 2 Polar Coded Link Simulation ===")
    print()

    configs = [
        {"rate_str": "1/2", "code_length": 256, "mod_type": "BPSK", "label": "Polar(256,112) R=1/2 BPSK"},
        {"rate_str": "1/2", "code_length": 512, "mod_type": "BPSK", "label": "Polar(512,224) R=1/2 BPSK"},
        {"rate_str": "3/4", "code_length": 256, "mod_type": "BPSK", "label": "Polar(256,176) R=3/4 BPSK"},
        {"rate_str": "1/4", "code_length": 256, "mod_type": "BPSK", "label": "Polar(256,48) R=1/4 BPSK"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg['label']}...")
        res = sim_polar_coded_psk_link(
            num_info_bits=5000,
            snr_range_db=snr_range,
            **{k: v for k, v in cfg.items() if k != "label"},
        )
        results.append((cfg["label"], res))
        # 打印部分结果
        for s, b, f in zip(res["snr_db"][::4], res["ber"][::4], res["fer"][::4], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    # 也运行无编码 BPSK 作为对比
    print(f"[{len(configs)+1}/{len(configs)+1}] Uncoded BPSK (reference)...")
    uncoded = sim_psk_link(num_data_bits=5000, mod_type="BPSK",
                           frame_type=3, snr_range_db=snr_range)

    # BER 曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    markers = ["o-", "s-", "^-", "d-"]
    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=3)

    uncoded_ber = [max(b, 1e-6) for b in uncoded["ber"]]
    ax1.semilogy(uncoded["snr_db"], uncoded_ber, "x--", label="Uncoded BPSK", markersize=3)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE PHY - Phase 2 BER")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=3)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE PHY - Phase 2 FER")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase2.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase2.png")


# ── Phase 3: 帧级端到端仿真 ──


def sim_frame_link(
    frame_type: int = 2,
    num_data_bits: int = 200,
    rate_str: str = "1/2",
    code_length: int = 256,
    pilot_interval: int = 4,
    snr_range_db: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    """完整帧级仿真: 帧组装 → Polar编码 → PSK调制(含导频) → AWGN → 解调 → 解码 → BER。

    :param frame_type: 2, 3, or 4.
    :param num_data_bits: 数据负载比特数.
    :param rate_str: Polar码率.
    :param code_length: Polar码长.
    :param pilot_interval: 导频插入间隔 (0, 4, 8, 16).
    :param snr_range_db: SNR扫描范围.
    :param seed: 随机种子.

    :returns: {"snr_db": [...], "ber": [...], "fer": [...]}
    """
    from nearlink_sdr.phy.frame import (
        FrameConfig,
        assemble_frame_bits,
        frame_to_symbols,
    )
    from nearlink_sdr.phy.pilot import remove_pilots

    if snr_range_db is None:
        snr_range_db = np.arange(-2, 10, 1)

    K = get_info_bit_count(rate_str, code_length)
    enc = PolarEncoder(code_length, K)
    dec = get_polar_decoder(code_length, K)

    rng = np.random.default_rng(seed)

    # 每帧的数据比特分成码块
    n_blocks = max(1, num_data_bits // K)

    mod_type = {2: "QPSK", 3: "QPSK", 4: "BPSK"}[frame_type]
    config = FrameConfig(
        frame_type=frame_type,
        pilot_interval=pilot_interval,
        mod_type=mod_type,
    )

    ber_list = []
    fer_list = []

    for snr in snr_range_db:
        total_bit_errors = 0
        total_bits = 0
        frame_errors = 0

        for _ in range(n_blocks):
            info = rng.integers(0, 2, size=K, dtype=np.int8)

            # Polar编码
            coded = enc.encode(info)

            # 调制 + 帧组装
            ctrl_bits = np.zeros(64 if frame_type == 2 else 256, dtype=np.int8)
            fields = assemble_frame_bits(ctrl_bits, coded, config)
            tx_symbols = frame_to_symbols(fields, config)

            # AWGN信道
            snr_linear = 10.0 ** (float(snr) / 10.0)
            R_val = K / code_length
            noise_var = 1.0 / (2.0 * R_val * snr_linear) if snr_linear > 0 else 1e10
            noise = rng.normal(0, np.sqrt(noise_var), size=len(tx_symbols)) + \
                    1j * rng.normal(0, np.sqrt(noise_var), size=len(tx_symbols))
            rx_symbols = tx_symbols + noise

            # 提取数据段(从末尾反推)
            data_syms_count = code_length // (2 if mod_type == "QPSK" else 1)
            if pilot_interval > 0:
                n_data_pilots = data_syms_count // pilot_interval
                if data_syms_count % pilot_interval == 0 and n_data_pilots > 0:
                    n_data_pilots -= 1  # 末尾导频省略
                data_section_len = data_syms_count + n_data_pilots
            else:
                data_section_len = data_syms_count
            data_start = len(tx_symbols) - data_section_len

            # 提取数据符号并去除导频
            data_rx = rx_symbols[data_start:]
            data_syms = remove_pilots(data_rx, pilot_interval) if pilot_interval > 0 else data_rx

            # 解调为LLR (BPSK/QPSK)
            if mod_type == "BPSK":
                # 取虚部 (bit 0→90°→imag=1, bit 1→-90°→imag=-1)
                llr = np.zeros(min(code_length, len(data_syms)))
                for i in range(len(llr)):
                    sym = data_syms[i]
                    if i % 2 == 1:
                        sym *= np.exp(1j * np.pi / 2)
                    llr[i] = np.imag(sym) * 2.0 / noise_var
            else:
                # QPSK: 2 bits per symbol
                llr = np.zeros(min(code_length, len(data_syms) * 2))
                for i in range(min(len(data_syms), code_length // 2)):
                    sym = data_syms[i]
                    if i % 2 == 1:
                        sym *= np.exp(1j * np.deg2rad(45.0))
                    # I/Q to LLR: b0→imag, b1→real
                    llr[2*i] = np.imag(sym) * 2.0 / noise_var
                    llr[2*i+1] = np.real(sym) * 2.0 / noise_var

            # 补齐到码长
            if len(llr) < code_length:
                llr = np.concatenate([llr, np.zeros(code_length - len(llr))])
            llr = llr[:code_length]

            # SC解码
            decoded = dec.decode(llr)

            bit_errors = int(np.sum(decoded != info))
            total_bit_errors += bit_errors
            total_bits += K
            if bit_errors > 0:
                frame_errors += 1

        ber = total_bit_errors / total_bits if total_bits > 0 else 0.0
        fer = frame_errors / n_blocks
        ber_list.append(ber)
        fer_list.append(fer)

    return {
        "snr_db": snr_range_db.tolist(),
        "ber": ber_list,
        "fer": fer_list,
    }


def run_phase3_simulation():
    """Phase 3 帧级仿真: 不同帧类型、导频配置的 BER/FER。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(-2, 8, 0.5)

    print("=== Phase 3 Frame-Level Simulation ===")
    print()

    configs = [
        {"frame_type": 2, "pilot_interval": 0, "label": "Type2 QPSK no-pilot"},
        {"frame_type": 2, "pilot_interval": 4, "label": "Type2 QPSK pilot=4"},
        {"frame_type": 3, "pilot_interval": 4, "label": "Type3 QPSK pilot=4"},
        {"frame_type": 4, "pilot_interval": 4, "label": "Type4 BPSK pilot=4"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg['label']}...")
        res = sim_frame_link(
            frame_type=cfg["frame_type"],
            num_data_bits=3000,
            rate_str="1/2",
            code_length=256,
            pilot_interval=cfg["pilot_interval"],
            snr_range_db=snr_range,
        )
        results.append((cfg["label"], res))
        for s, b, f in zip(res["snr_db"][::4], res["ber"][::4], res["fer"][::4], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    markers = ["o-", "s-", "^-", "d-"]

    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=3)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE - Phase 3 Frame-Level BER")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=3)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE - Phase 3 Frame-Level FER")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase3.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase3.png")


# ── Phase 4: 多径信道 + 均衡仿真 ──


def sim_channel_eq_link(
    channel_type: str = "rayleigh",
    rician_k_db: float = 6.0,
    eq_method: str = "mmse",
    rate_str: str = "1/2",
    code_length: int = 256,
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 20,
    seed: int = 42,
) -> dict:
    """多径信道 + 均衡器 + Polar编码 BPSK 链路仿真。

    流程: 信息比特 → Polar编码 → BPSK调制 → 信道(衰落+AWGN) → 均衡 → LLR → 解码

    :param channel_type: "awgn" / "rayleigh" / "rician" / "multipath"。
    :param rician_k_db: Rician K 因子。
    :param eq_method: "zf" / "mmse" / "none" (不均衡)。
    :param rate_str: Polar 码率。
    :param code_length: Polar 码长。
    :param snr_range_db: SNR 范围。
    :param n_frames: 每个 SNR 点的帧数。
    :param seed: 随机种子。

    :returns: {"snr_db": [...], "ber": [...], "fer": [...]}
    """
    from nearlink_sdr.phy.channel import PDP_2TAP, ChannelConfig
    from nearlink_sdr.phy.equalizer import equalize_1tap, equalize_mmse_freq

    if snr_range_db is None:
        snr_range_db = np.arange(-2, 12, 1)

    K = get_info_bit_count(rate_str, code_length)
    enc = PolarEncoder(code_length, K)
    dec = get_polar_decoder(code_length, K)

    rng = np.random.default_rng(seed)
    ber_list, fer_list = [], []

    for snr in snr_range_db:
        total_errors, total_bits, frame_errors = 0, 0, 0

        for _ in range(n_frames):
            info = rng.integers(0, 2, size=K, dtype=np.int8)
            coded = enc.encode(info)

            # BPSK 调制: 0 → +1, 1 → -1
            tx = (1 - 2 * coded.astype(np.float64)).astype(complex)

            # 信道
            frame_seed = int(rng.integers(0, 2**31))
            cfg = ChannelConfig(
                snr_db=float(snr),
                channel_type=channel_type,
                rician_k_db=rician_k_db,
                pdp=list(PDP_2TAP),
                seed=frame_seed,
            )
            ch = ChannelModel(config=cfg)
            noise_var = ch.noise_variance

            if channel_type == "multipath":
                taps = ch.get_channel_taps(len(tx))
                # 手动应用多径: 对每条路径做延迟卷积
                n_sig = len(tx)
                faded = np.zeros(n_sig, dtype=complex)
                for tap_i, (delay, _) in enumerate(PDP_2TAP):
                    if delay < n_sig and tap_i < taps.shape[0]:
                        shifted = np.zeros(n_sig, dtype=complex)
                        shifted[delay:] = tx[:n_sig - delay]
                        faded += taps[tap_i, :n_sig] * shifted
                # 手动加噪
                sig_power = np.mean(np.abs(faded) ** 2) if np.mean(np.abs(faded) ** 2) > 1e-20 else 1.0
                n_power = sig_power / (10.0 ** (float(snr) / 10.0)) if snr > -50 else sig_power * 100
                noise = np.sqrt(n_power / 2) * (ch._rng.standard_normal(n_sig) + 1j * ch._rng.standard_normal(n_sig))
                rx = faded + noise

                if eq_method == "none":
                    eq = rx
                else:
                    h_time = taps[:, 0]
                    h_freq = np.fft.fft(h_time, len(rx))
                    if eq_method == "mmse":
                        eq = equalize_mmse_freq(rx, h_freq, noise_var)
                    else:
                        from nearlink_sdr.phy.equalizer import equalize_zf
                        eq = equalize_zf(rx, h_freq)
            elif channel_type in ("rayleigh", "rician"):
                taps = ch.get_channel_taps(len(tx))
                h = taps[0, :]
                # 手动应用平坦衰落 + 加噪
                faded = tx * h
                sig_power = np.mean(np.abs(faded) ** 2) if np.mean(np.abs(faded) ** 2) > 1e-20 else 1.0
                n_power = sig_power / (10.0 ** (float(snr) / 10.0)) if snr > -50 else sig_power * 100
                noise = np.sqrt(n_power / 2) * (ch._rng.standard_normal(len(tx)) + 1j * ch._rng.standard_normal(len(tx)))
                rx = faded + noise

                if eq_method == "none":
                    eq = rx
                else:
                    eq = equalize_1tap(rx, h, noise_var=noise_var, method=eq_method)
            else:
                rx = ch.apply_fading(tx)
                eq = rx

            # 软 LLR: BPSK +1→bit0, -1→bit1, LLR = 2*real(y)/sigma^2
            noise_var = ch.noise_variance
            if noise_var < 1e-10:
                noise_var = 1e-10
            llr = np.real(eq) * 2.0 / noise_var
            llr = llr[:code_length]
            if len(llr) < code_length:
                llr = np.concatenate([llr, np.zeros(code_length - len(llr))])

            decoded = dec.decode(llr.real)
            errors = int(np.sum(decoded != info))
            total_errors += errors
            total_bits += K
            if errors > 0:
                frame_errors += 1

        ber = total_errors / total_bits if total_bits > 0 else 0.0
        fer = frame_errors / n_frames
        ber_list.append(ber)
        fer_list.append(fer)

    return {"snr_db": snr_range_db.tolist(), "ber": ber_list, "fer": fer_list}


def run_phase4_simulation():
    """Phase 4: 不同信道模型 + 均衡器的 BER/FER 比较。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(-2, 12, 1)

    print("=== Phase 4 Channel & Equalizer Simulation ===")
    print()

    configs = [
        {"channel_type": "awgn", "eq_method": "none", "label": "AWGN (baseline)"},
        {"channel_type": "rayleigh", "eq_method": "none", "label": "Rayleigh (no eq)"},
        {"channel_type": "rayleigh", "eq_method": "mmse", "label": "Rayleigh + MMSE eq"},
        {"channel_type": "rician", "eq_method": "none", "label": "Rician K=6dB (no eq)"},
        {"channel_type": "rician", "eq_method": "mmse", "label": "Rician K=6dB + MMSE eq"},
        {"channel_type": "multipath", "eq_method": "none", "label": "Multipath (no eq)"},
        {"channel_type": "multipath", "eq_method": "mmse", "label": "Multipath + MMSE eq"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg['label']}...")
        res = sim_channel_eq_link(
            channel_type=cfg["channel_type"],
            eq_method=cfg["eq_method"],
            rate_str="1/2",
            code_length=256,
            snr_range_db=snr_range,
            n_frames=30,
        )
        results.append((cfg["label"], res))
        for s, b, f in zip(res["snr_db"][::3], res["ber"][::3], res["fer"][::3], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    markers = ["o-", "s--", "s-", "^--", "^-", "d--", "d-"]

    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=3)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE - Phase 4 Channel BER")
    ax1.legend(fontsize=6)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=3)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE - Phase 4 Channel FER")
    ax2.legend(fontsize=6)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase4.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase4.png")


# ── Phase 5: 跳频链路仿真 ──


def sim_hopping_link(
    hop_param2: int = 0xABCD,
    n_hops: int = 20,
    rate_str: str = "1/2",
    code_length: int = 256,
    snr_range_db: np.ndarray | None = None,
    bandwidth_mhz: int = 1,
    blocked_ratio: float = 0.0,
    seed: int = 42,
) -> dict:
    """跳频链路仿真: 数据帧在多个跳频信道间传输, 每跳独立衰落。

    流程: 生成跳频序列 → 每跳: Polar编码 → BPSK调制 → 独立 Rayleigh 信道 → 均衡 → 解码

    :param hop_param2: 跳频参数 2。
    :param n_hops: 每个 SNR 点的跳频次数 (帧数)。
    :param rate_str: Polar 码率。
    :param code_length: Polar 码长。
    :param snr_range_db: SNR 范围。
    :param bandwidth_mhz: 信道带宽 1/2/4 MHz。
    :param blocked_ratio: 被阻塞信道占比 (0~1)。
    :param seed: 随机种子。

    :returns: {"snr_db": [...], "ber": [...], "fer": [...], "channels_used": [...]}
    """
    from nearlink_sdr.phy.channel import ChannelConfig
    from nearlink_sdr.phy.equalizer import equalize_1tap
    from nearlink_sdr.phy.freq_hopping import (
        FreqTable,
        generate_hopping_sequence,
    )

    if snr_range_db is None:
        snr_range_db = np.arange(-2, 14, 2)

    rng = np.random.default_rng(seed)
    K = get_info_bit_count(rate_str, code_length)
    enc = PolarEncoder(code_length, K)
    dec = get_polar_decoder(code_length, K)

    # 频率表: 按比例阻塞部分信道
    ft = FreqTable(band="2400", bandwidth_mhz=bandwidth_mhz)
    all_ch = ft.full_table()
    n_blocked = int(len(all_ch) * blocked_ratio)
    if n_blocked > 0:
        blocked_idx = rng.choice(len(all_ch), n_blocked, replace=False)
        ft.blocked_channels = {all_ch[i] for i in blocked_idx}

    # 生成跳频序列
    hop_seq = generate_hopping_sequence(
        n_hops, hop_param2, ft, link_type="data", start_slot=0,
    )

    ber_list, fer_list = [], []
    channels_used = sorted(set(hop_seq))

    for snr in snr_range_db:
        total_errors, total_bits, frame_errors = 0, 0, 0

        for _hop_i, _ch_num in enumerate(hop_seq):
            info = rng.integers(0, 2, size=K, dtype=np.int8)
            coded = enc.encode(info)
            tx = (1 - 2 * coded.astype(np.float64)).astype(complex)

            # 每跳独立 Rayleigh 衰落 (模拟跳频分集效果)
            frame_seed = int(rng.integers(0, 2**31))
            cfg = ChannelConfig(
                snr_db=float(snr),
                channel_type="rayleigh",
                seed=frame_seed,
            )
            ch = ChannelModel(config=cfg)
            noise_var = ch.noise_variance

            taps = ch.get_channel_taps(len(tx))
            h = taps[0, :]
            faded = tx * h
            sig_power = max(np.mean(np.abs(faded) ** 2), 1e-20)
            n_power = sig_power / (10.0 ** (float(snr) / 10.0))
            noise = np.sqrt(n_power / 2) * (
                ch._rng.standard_normal(len(tx))
                + 1j * ch._rng.standard_normal(len(tx))
            )
            rx = faded + noise

            eq = equalize_1tap(rx, h, noise_var=noise_var, method="mmse")

            llr = np.real(eq) * 2.0 / max(noise_var, 1e-10)
            llr = llr[:code_length]
            decoded = dec.decode(llr.real)
            errors = int(np.sum(decoded != info))
            total_errors += errors
            total_bits += K
            if errors > 0:
                frame_errors += 1

        ber = total_errors / total_bits if total_bits > 0 else 0.0
        fer = frame_errors / n_hops
        ber_list.append(ber)
        fer_list.append(fer)

    return {
        "snr_db": snr_range_db.tolist(),
        "ber": ber_list,
        "fer": fer_list,
        "channels_used": channels_used,
    }


def run_phase5_simulation():
    """Phase 5: 跳频链路仿真 — 比较不同跳频配置的 BER/FER。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nearlink_sdr.phy.freq_hopping import derive_hop_param2

    snr_range = np.arange(-2, 14, 2)

    print("=== Phase 5 Frequency Hopping Link Simulation ===")
    print()

    hp2 = derive_hop_param2(0xDEADBEEF, 32)

    configs = [
        {"bandwidth_mhz": 1, "blocked_ratio": 0.0,
         "label": "1 MHz, no blocking"},
        {"bandwidth_mhz": 1, "blocked_ratio": 0.3,
         "label": "1 MHz, 30% blocked"},
        {"bandwidth_mhz": 2, "blocked_ratio": 0.0,
         "label": "2 MHz, no blocking"},
        {"bandwidth_mhz": 4, "blocked_ratio": 0.0,
         "label": "4 MHz, no blocking"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {cfg['label']}...")
        res = sim_hopping_link(
            hop_param2=hp2,
            n_hops=30,
            rate_str="1/2",
            code_length=256,
            snr_range_db=snr_range,
            bandwidth_mhz=cfg["bandwidth_mhz"],
            blocked_ratio=cfg["blocked_ratio"],
        )
        results.append((cfg["label"], res))
        print(f"  Channels used: {len(res['channels_used'])}")
        for s, b, f in zip(res["snr_db"][::2], res["ber"][::2], res["fer"][::2], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    markers = ["o-", "s--", "^-", "d-"]

    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=4)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE - Phase 5 Hopping BER")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=4)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE - Phase 5 Hopping FER")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase5.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase5.png")


# ── Phase 6: 全链路 Pipeline 仿真 ──


def sim_pipeline_link(
    frame_type: int = 2,
    mcs_index: int = 7,
    n_data_bytes: int = 10,
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 20,
    sps: int = 4,
    seed: int = 42,
) -> dict:
    """使用 tx_chain / rx_chain 的全链路端到端仿真。

    :param frame_type: 帧类型 (1-4)
    :param mcs_index: MCS 索引
    :param n_data_bytes: 数据长度 (字节)
    :param snr_range_db: Eb/N0 扫描范围 (dB)
    :param n_frames: 每个 SNR 点的仿真帧数
    :param sps: 每符号采样数
    :param seed: 随机种子

    :returns: {"snr_db": [...], "ber": [...], "fer": [...]}
    """
    from nearlink_sdr.phy.rx_pipeline import rx_chain
    from nearlink_sdr.phy.tx_pipeline import TxConfig, tx_chain

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    # 根据帧类型确定配置
    if frame_type == 1:
        ctrl_bits_len = 20
        crc_len = 24
        head_crc_len = 12
        pilot_interval = 0
    elif frame_type == 2:
        ctrl_bits_len = 28
        crc_len = 24
        head_crc_len = 12
        pilot_interval = 8
    else:  # FT3/FT4
        ctrl_bits_len = 27
        crc_len = 24
        head_crc_len = 24
        pilot_interval = 4

    cfg = TxConfig(
        frame_type=frame_type,
        mcs_index=mcs_index,
        pid=0x123456 if frame_type <= 2 else 0,  # FT3/4 用 m_seq_index
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=crc_len,
        ctrl_bits_len=ctrl_bits_len,
        pilot_interval=pilot_interval,
        sps=sps,
    )

    n_data_bits = n_data_bytes * 8
    ber_list, fer_list = [], []

    for snr in snr_range_db:
        total_errors, total_bits, frame_errors = 0, 0, 0
        ch = ChannelModel(snr_db=float(snr))

        for _ in range(n_frames):
            # 生成随机控制信息和数据
            head_bits = rng.integers(0, 2, ctrl_bits_len + head_crc_len, dtype=np.int8)
            data_bits = rng.integers(0, 2, n_data_bits, dtype=int)

            # TX
            iq = tx_chain(head_bits, data_bits, cfg)

            # AWGN 信道
            rx_iq = ch.apply_awgn(iq, sps)

            # RX
            result = rx_chain(rx_iq, cfg, n_data_bytes)

            # 统计
            errors = int(np.sum(result.data_bits[:n_data_bits] != data_bits))
            total_errors += errors
            total_bits += n_data_bits
            if not result.crc_ok:
                frame_errors += 1

        ber = total_errors / total_bits if total_bits > 0 else 0.0
        fer = frame_errors / n_frames
        ber_list.append(ber)
        fer_list.append(fer)

    return {"snr_db": snr_range_db.tolist(), "ber": ber_list, "fer": fer_list}


def run_phase6_simulation():
    """Phase 6: 全链路 Pipeline BER/FER 仿真 — 不同 MCS + 帧类型。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(0, 16, 1)

    print("=== Phase 6 Full Pipeline Link Simulation ===")
    print()

    configs = [
        {"frame_type": 2, "mcs_index": 5, "label": "FT2 MCS5 QPSK R=5/8"},
        {"frame_type": 2, "mcs_index": 7, "label": "FT2 MCS7 QPSK R=7/8"},
        {"frame_type": 2, "mcs_index": 8, "label": "FT2 MCS8 QPSK R=1/1"},
        {"frame_type": 4, "mcs_index": 0, "label": "FT4 MCS0 BPSK R=1/4"},
        {"frame_type": 1, "mcs_index": 8, "label": "FT1 GFSK uncoded"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg['label']}...")
        res = sim_pipeline_link(
            frame_type=cfg["frame_type"],
            mcs_index=cfg["mcs_index"],
            n_data_bytes=10,
            snr_range_db=snr_range,
            n_frames=50,
        )
        results.append((cfg["label"], res))
        for s, b, f in zip(res["snr_db"][::4], res["ber"][::4], res["fer"][::4], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    markers = ["o-", "s-", "^-", "d-", "x-"]

    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=4)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE - Phase 6 Pipeline BER")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=4)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE - Phase 6 Pipeline FER")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase6.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase6.png")


# ── Phase 7: 多径信道 + 频率偏移 Pipeline 仿真 ──


def _apply_cfo(signal: np.ndarray, cfo_hz: float, sample_rate: float) -> np.ndarray:
    """对 IQ 信号施加载波频率偏移。"""
    if cfo_hz == 0.0:
        return signal
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * cfo_hz * t)


def sim_pipeline_channel_link(
    frame_type: int = 2,
    mcs_index: int = 7,
    n_data_bytes: int = 10,
    channel_type: str = "awgn",
    rician_k_db: float = 6.0,
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 20,
    sps: int = 4,
    seed: int = 42,
) -> dict:
    """全链路 Pipeline 仿真 — 支持多径信道、频率偏移和均衡器。

    在 tx_chain 输出的 IQ 信号上依次施加:
      1. 衰落信道 (Rayleigh / Rician / 多径)
      2. 载波频率偏移
      3. AWGN 噪声
    然后可选地进行均衡, 最后送入 rx_chain 解码。

    :param frame_type: 帧类型 (1-4)。
    :param mcs_index: MCS 索引。
    :param n_data_bytes: 数据长度 (字节)。
    :param channel_type: "awgn" | "rayleigh" | "rician" | "multipath"。
    :param rician_k_db: Rician K 因子 (dB)。
    :param cfo_hz: 载波频率偏移 (Hz)。
    :param eq_method: "none" | "zf" | "mmse", 仅对衰落信道有效。
    :param snr_range_db: Eb/N0 扫描范围。
    :param n_frames: 每个 SNR 点的仿真帧数。
    :param sps: 每符号采样数。
    :param seed: 随机种子。

    :returns: {"snr_db": [...], "ber": [...], "fer": [...]}
    """
    from nearlink_sdr.phy.channel import ChannelConfig
    from nearlink_sdr.phy.equalizer import equalize_1tap, equalize_mmse_freq
    from nearlink_sdr.phy.rx_pipeline import rx_chain
    from nearlink_sdr.phy.tx_pipeline import TxConfig, tx_chain

    if snr_range_db is None:
        snr_range_db = np.arange(0, 20, 2)

    rng = np.random.default_rng(seed)

    # SLE 符号速率 1 Msps, 采样率 = sps * 符号速率
    sample_rate = sps * 1e6

    # 帧类型参数
    if frame_type == 1:
        ctrl_bits_len, crc_len, head_crc_len, pilot_interval = 20, 24, 12, 0
    elif frame_type == 2:
        ctrl_bits_len, crc_len, head_crc_len, pilot_interval = 28, 24, 12, 8
    else:
        ctrl_bits_len, crc_len, head_crc_len, pilot_interval = 27, 24, 24, 4

    cfg = TxConfig(
        frame_type=frame_type,
        mcs_index=mcs_index,
        pid=0x123456 if frame_type <= 2 else 0,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=crc_len,
        ctrl_bits_len=ctrl_bits_len,
        pilot_interval=pilot_interval,
        sps=sps,
    )

    n_data_bits = n_data_bytes * 8
    ber_list, fer_list = [], []

    for snr in snr_range_db:
        total_errors, total_bits, frame_errors = 0, 0, 0

        for _ in range(n_frames):
            head_bits = rng.integers(0, 2, ctrl_bits_len + head_crc_len, dtype=np.int8)
            data_bits = rng.integers(0, 2, n_data_bits, dtype=int)

            iq = tx_chain(head_bits, data_bits, cfg)

            # 施加信道损伤
            frame_seed = int(rng.integers(0, 2**31))
            ch_cfg = ChannelConfig(
                snr_db=float(snr),
                channel_type=channel_type,
                rician_k_db=rician_k_db,
                seed=frame_seed,
            )
            ch = ChannelModel(config=ch_cfg)

            rx_iq = ch.apply_awgn(iq, sps) if channel_type == "awgn" else ch.apply_fading(iq, sps)

            # 施加频率偏移
            rx_iq = _apply_cfo(rx_iq, cfo_hz, sample_rate)

            # 均衡 (genie-aided: 使用 apply_fading 缓存的信道系数)
            if eq_method != "none" and channel_type != "awgn":
                noise_var = ch.noise_variance
                taps = ch.last_taps
                if channel_type in ("rayleigh", "rician"):
                    h = taps[0, :]
                    rx_iq = equalize_1tap(rx_iq, h, noise_var, method=eq_method)
                elif channel_type == "multipath":
                    h_time = taps[:, 0]
                    h_freq = np.fft.fft(h_time, len(rx_iq))
                    rx_iq = equalize_mmse_freq(rx_iq, h_freq, noise_var)

            result = rx_chain(rx_iq, cfg, n_data_bytes)

            errors = int(np.sum(result.data_bits[:n_data_bits] != data_bits))
            total_errors += errors
            total_bits += n_data_bits
            if not result.crc_ok:
                frame_errors += 1

        ber = total_errors / total_bits if total_bits > 0 else 0.0
        fer = frame_errors / n_frames
        ber_list.append(ber)
        fer_list.append(fer)

    return {"snr_db": snr_range_db.tolist(), "ber": ber_list, "fer": fer_list}


def run_phase7_simulation():
    """Phase 7: 多径信道 + 频率偏移 Pipeline BER/FER 仿真。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(0, 20, 1)

    print("=== Phase 7 Channel Impairment Pipeline Simulation ===")
    print()

    configs = [
        {"channel_type": "awgn", "cfo_hz": 0.0, "eq_method": "none",
         "label": "AWGN (baseline)"},
        {"channel_type": "rayleigh", "cfo_hz": 0.0, "eq_method": "none",
         "label": "Rayleigh (no eq)"},
        {"channel_type": "rayleigh", "cfo_hz": 0.0, "eq_method": "mmse",
         "label": "Rayleigh + MMSE eq"},
        {"channel_type": "rician", "cfo_hz": 0.0, "eq_method": "none",
         "label": "Rician K=6dB (no eq)"},
        {"channel_type": "rician", "cfo_hz": 0.0, "eq_method": "mmse",
         "label": "Rician K=6dB + MMSE eq"},
        {"channel_type": "awgn", "cfo_hz": 500.0, "eq_method": "none",
         "label": "AWGN + CFO 500Hz"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg['label']}...")
        res = sim_pipeline_channel_link(
            frame_type=2,
            mcs_index=7,
            n_data_bytes=10,
            channel_type=cfg["channel_type"],
            cfo_hz=cfg["cfo_hz"],
            eq_method=cfg["eq_method"],
            snr_range_db=snr_range,
            n_frames=50,
        )
        results.append((cfg["label"], res))
        for s, b, f in zip(res["snr_db"][::5], res["ber"][::5], res["fer"][::5], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    markers = ["o-", "s-", "^-", "d--", "x--", "v-"]

    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=4)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE - Phase 7 Channel Impairment BER (FT2 MCS7)")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=4)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE - Phase 7 Channel Impairment FER (FT2 MCS7)")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase7.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase7.png")


# ── Phase 8: 多帧类型 + 信道 + 均衡器综合仿真 ──


def run_phase8_simulation():
    """Phase 8: 多帧类型在不同信道条件下的 BER/FER 综合对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(0, 20, 1)

    print("=== Phase 8 Multi-Frame-Type Channel Simulation ===")
    print()

    configs = [
        # AWGN 基线: 各帧类型
        {"frame_type": 1, "mcs_index": 8, "channel_type": "awgn",
         "eq_method": "none", "label": "FT1 GFSK AWGN"},
        {"frame_type": 2, "mcs_index": 5, "channel_type": "awgn",
         "eq_method": "none", "label": "FT2 MCS5 R=5/8 AWGN"},
        {"frame_type": 2, "mcs_index": 7, "channel_type": "awgn",
         "eq_method": "none", "label": "FT2 MCS7 R=7/8 AWGN"},
        {"frame_type": 3, "mcs_index": 0, "channel_type": "awgn",
         "eq_method": "none", "label": "FT3 MCS0 R=1/4 AWGN"},
        {"frame_type": 4, "mcs_index": 0, "channel_type": "awgn",
         "eq_method": "none", "label": "FT4 MCS0 R=1/4 AWGN"},
        # Rayleigh + MMSE 均衡
        {"frame_type": 2, "mcs_index": 7, "channel_type": "rayleigh",
         "eq_method": "mmse", "label": "FT2 MCS7 Rayleigh+MMSE"},
        {"frame_type": 3, "mcs_index": 0, "channel_type": "rayleigh",
         "eq_method": "mmse", "label": "FT3 MCS0 Rayleigh+MMSE"},
        {"frame_type": 4, "mcs_index": 0, "channel_type": "rayleigh",
         "eq_method": "mmse", "label": "FT4 MCS0 Rayleigh+MMSE"},
    ]

    results = []
    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg['label']}...")
        res = sim_pipeline_channel_link(
            frame_type=cfg["frame_type"],
            mcs_index=cfg["mcs_index"],
            n_data_bytes=10,
            channel_type=cfg["channel_type"],
            eq_method=cfg["eq_method"],
            snr_range_db=snr_range,
            n_frames=50,
        )
        results.append((cfg["label"], res))
        for s, b, f in zip(res["snr_db"][::5], res["ber"][::5], res["fer"][::5], strict=False):
            print(f"  Eb/N0={s:5.1f} dB  BER={b:.5f}  FER={f:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    markers = ["o-", "s-", "^-", "d-", "x-", "^--", "d--", "x--"]

    for (label, res), mk in zip(results, markers, strict=False):
        ber_plot = [max(b, 1e-6) for b in res["ber"]]
        ax1.semilogy(res["snr_db"], ber_plot, mk, label=label, markersize=4)
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate")
    ax1.set_title("SparkLink SLE - Phase 8 Multi-FT BER")
    ax1.legend(fontsize=6)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim(bottom=1e-5)

    for (label, res), mk in zip(results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax2.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=4)
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Frame Error Rate")
    ax2.set_title("SparkLink SLE - Phase 8 Multi-FT FER")
    ax2.legend(fontsize=6)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=1e-4)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase8.png", dpi=150)
    print("\nBER/FER curves saved to ber_phase8.png")


# ── Phase 9: MAC 帧级端到端仿真 ──


def _channel_impair(
    iq: np.ndarray,
    snr_db: float,
    channel_type: str,
    rician_k_db: float,
    cfo_hz: float,
    eq_method: str,
    sps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """对 IQ 信号施加信道损伤 (衰落/噪声/频偏) 并可选均衡。"""
    from nearlink_sdr.phy.channel import ChannelConfig
    from nearlink_sdr.phy.equalizer import equalize_1tap, equalize_mmse_freq

    frame_seed = int(rng.integers(0, 2**31))
    ch_cfg = ChannelConfig(
        snr_db=snr_db,
        channel_type=channel_type,
        rician_k_db=rician_k_db,
        seed=frame_seed,
    )
    ch = ChannelModel(config=ch_cfg)

    rx_iq = ch.apply_awgn(iq, sps) if channel_type == "awgn" else ch.apply_fading(iq, sps)

    if cfo_hz != 0.0:
        sample_rate = sps * 1e6
        rx_iq = _apply_cfo(rx_iq, cfo_hz, sample_rate)

    if eq_method != "none" and channel_type != "awgn":
        noise_var = ch.noise_variance
        taps = ch.last_taps
        if channel_type in ("rayleigh", "rician"):
            h = taps[0, :]
            rx_iq = equalize_1tap(rx_iq, h, noise_var, method=eq_method)
        elif channel_type == "multipath":
            h_time = taps[:, 0]
            h_freq = np.fft.fft(h_time, len(rx_iq))
            rx_iq = equalize_mmse_freq(rx_iq, h_freq, noise_var)

    return rx_iq


def sim_mac_signaling_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """MAC 信令帧端到端仿真: 信令编码 → IQ → 信道 → IQ → 信令解码。

    每帧随机选取一种信令类型进行编码,
    经过 PHY 发射/接收流水线和信道后, 统计信令解码成功率。

    :param channel_type: "awgn" | "rayleigh" | "rician" | "multipath"。
    :param cfo_hz: 载波频率偏移 (Hz)。
    :param eq_method: "none" | "zf" | "mmse"。
    :param rician_k_db: Rician K 因子 (dB)。

    :returns: {"snr_db": [...], "signaling_success_rate": [...]}
    """
    from nearlink_sdr.mac.link_control import (
        IntervalUpdateRequest,
        PingRequest,
        PingResponse,
        TimeoutUpdateRequest,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_signaling, signaling_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    # 构造信令池
    def _make_msg(idx: int):
        choices = [
            PingRequest(),
            PingResponse(),
            IntervalUpdateRequest(interval_type=int(rng.integers(0, 8))),
            TimeoutUpdateRequest(timeout=int(rng.integers(1, 1000))),
        ]
        return choices[idx % len(choices)]

    from nearlink_sdr.mac.signaling import encode_signaling

    success_rates = []

    for snr in snr_range_db:
        successes = 0

        for i in range(n_frames):
            msg = _make_msg(i)
            frame = encode_signaling(msg)
            mac_bytes = frame.pack()
            n_mac_bytes = len(mac_bytes)

            iq = signaling_to_iq(msg, cfg)
            rx_iq = _channel_impair(
                iq, float(snr), channel_type, rician_k_db,
                cfo_hz, eq_method, cfg.sps, rng,
            )

            recovered, ok = iq_to_signaling(rx_iq, cfg, n_mac_bytes)
            if (
                ok
                and recovered is not None
                and type(recovered).__name__ == type(msg).__name__
            ):
                successes += 1

        success_rates.append(successes / n_frames)

    return {"snr_db": snr_range_db.tolist(), "signaling_success_rate": success_rates}


def sim_mac_data_link(
    payload_sizes: list[int] | None = None,
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """MAC 异步数据帧端到端仿真: 数据编码 → IQ → 信道 → IQ → 数据解码。

    对不同载荷大小, 统计字节级误码率和帧正确率。

    :param payload_sizes: 要测试的载荷大小列表 (字节)。
    :param channel_type: "awgn" | "rayleigh" | "rician" | "multipath"。
    :param cfo_hz: 载波频率偏移 (Hz)。
    :param eq_method: "none" | "zf" | "mmse"。
    :param rician_k_db: Rician K 因子 (dB)。

    :returns: {"snr_db": [...], "results": {size: {"fer": [...], "byte_ber": [...]}}}
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if payload_sizes is None:
        payload_sizes = [4, 10, 27]
    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    all_results: dict = {}

    for size in payload_sizes:
        fer_list, byte_ber_list = [], []

        for snr in snr_range_db:
            frame_errors, total_byte_errors, total_bytes = 0, 0, 0

            for _ in range(n_frames):
                payload = bytes(rng.integers(0, 256, size, dtype=np.uint8))
                frame = AsyncDataFrame(segment_type=0, data=payload)
                mac_bytes = frame.pack()
                n_mac_bytes = len(mac_bytes)

                iq = mac_to_iq(mac_bytes, cfg)
                rx_iq = _channel_impair(
                    iq, float(snr), channel_type, rician_k_db,
                    cfo_hz, eq_method, cfg.sps, rng,
                )
                rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

                if not rx.crc_ok:
                    frame_errors += 1
                    total_byte_errors += size
                else:
                    try:
                        recovered = AsyncDataFrame.unpack(rx.mac_payload)
                        errors = sum(
                            a != b
                            for a, b in zip(payload, recovered.data, strict=False)
                        )
                        total_byte_errors += errors
                    except (ValueError, IndexError):
                        frame_errors += 1
                        total_byte_errors += size

                total_bytes += size

            fer_list.append(frame_errors / n_frames)
            byte_ber_list.append(
                total_byte_errors / total_bytes if total_bytes > 0 else 0.0
            )

        all_results[size] = {"fer": fer_list, "byte_ber": byte_ber_list}

    return {"snr_db": snr_range_db.tolist(), "results": all_results}


def sim_mac_mux_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    data_size: int = 10,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """MAC 复用帧端到端仿真: 控制帧+数据帧复用 → IQ → 信道 → IQ → 解码。

    每帧包含一条信令和一段数据, 验证复用帧在信道传输后的完整性。

    :param channel_type: "awgn" | "rayleigh" | "rician" | "multipath"。
    :param cfo_hz: 载波频率偏移 (Hz)。
    :param eq_method: "none" | "zf" | "mmse"。
    :param rician_k_db: Rician K 因子 (dB)。

    :returns: {"snr_db": [...], "mux_success_rate": [...], "data_match_rate": [...]}
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame, MuxFrame
    from nearlink_sdr.mac.link_control import PingRequest
    from nearlink_sdr.mac.signaling import encode_signaling
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    mux_success_list, data_match_list = [], []

    for snr in snr_range_db:
        mux_ok_count, data_ok_count = 0, 0

        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, data_size, dtype=np.uint8))
            ctrl_frame = encode_signaling(PingRequest())
            data_frame = AsyncDataFrame(segment_type=0, data=payload)
            mux = MuxFrame(control_frames=[ctrl_frame], data_frame=data_frame)
            mac_bytes = mux.pack()
            n_mac_bytes = len(mac_bytes)

            iq = mac_to_iq(mac_bytes, cfg)
            rx_iq = _channel_impair(
                iq, float(snr), channel_type, rician_k_db,
                cfo_hz, eq_method, cfg.sps, rng,
            )
            rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

            if rx.crc_ok:
                mux_ok_count += 1
                # 验证数据部分: 复用帧中数据帧位于控制帧之后
                ctrl_len = len(ctrl_frame.pack())
                data_part = rx.mac_payload[ctrl_len:]
                try:
                    recovered = AsyncDataFrame.unpack(data_part)
                    if recovered.data == payload:
                        data_ok_count += 1
                except (ValueError, IndexError):
                    pass

        mux_success_list.append(mux_ok_count / n_frames)
        data_match_list.append(data_ok_count / n_frames)

    return {
        "snr_db": snr_range_db.tolist(),
        "mux_success_rate": mux_success_list,
        "data_match_rate": data_match_list,
    }


def run_phase9_simulation():
    """Phase 9: MAC 帧级端到端仿真 — 信令、数据、复用帧 (含信道损伤)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(0, 16, 1)

    print("=== Phase 9 MAC Frame-Level Simulation ===")
    print()

    # ── 信令帧: AWGN vs Rayleigh vs Rayleigh+MMSE ──
    sig_configs = [
        {"channel_type": "awgn", "eq_method": "none", "cfo_hz": 0.0,
         "label": "AWGN"},
        {"channel_type": "rayleigh", "eq_method": "none", "cfo_hz": 0.0,
         "label": "Rayleigh (no eq)"},
        {"channel_type": "rayleigh", "eq_method": "mmse", "cfo_hz": 0.0,
         "label": "Rayleigh + MMSE"},
        {"channel_type": "awgn", "eq_method": "none", "cfo_hz": 500.0,
         "label": "AWGN + CFO 500Hz"},
    ]
    print("[1/3] MAC signaling frame simulation...")
    sig_results = []
    for cfg in sig_configs:
        print(f"  {cfg['label']}...")
        res = sim_mac_signaling_link(
            snr_range_db=snr_range, n_frames=100,
            channel_type=cfg["channel_type"],
            eq_method=cfg["eq_method"],
            cfo_hz=cfg["cfo_hz"],
        )
        sig_results.append((cfg["label"], res))
        for s, r in zip(res["snr_db"][::4], res["signaling_success_rate"][::4],
                        strict=False):
            print(f"    SNR={s:2.0f} dB  Success={r:.3f}")

    # ── 数据帧: AWGN vs Rayleigh+MMSE ──
    print("[2/3] MAC data frame simulation...")
    data_configs = [
        {"channel_type": "awgn", "eq_method": "none", "label": "AWGN"},
        {"channel_type": "rayleigh", "eq_method": "mmse", "label": "Rayleigh+MMSE"},
    ]
    data_results = []
    for dcfg in data_configs:
        print(f"  {dcfg['label']}...")
        res = sim_mac_data_link(
            payload_sizes=[4, 10, 27],
            snr_range_db=snr_range, n_frames=100,
            channel_type=dcfg["channel_type"],
            eq_method=dcfg["eq_method"],
        )
        data_results.append((dcfg["label"], res))
        for size, metrics in res["results"].items():
            for s, f in zip(res["snr_db"][::4], metrics["fer"][::4], strict=False):
                print(f"    {dcfg['label']} {size}B  SNR={s:2.0f} dB  FER={f:.3f}")

    # ── 复用帧: AWGN vs Rayleigh+MMSE ──
    print("[3/3] MAC mux frame simulation...")
    mux_configs = [
        {"channel_type": "awgn", "eq_method": "none", "label": "AWGN"},
        {"channel_type": "rayleigh", "eq_method": "mmse", "label": "Rayleigh+MMSE"},
    ]
    mux_results = []
    for mcfg in mux_configs:
        print(f"  {mcfg['label']}...")
        res = sim_mac_mux_link(
            snr_range_db=snr_range, n_frames=100, data_size=10,
            channel_type=mcfg["channel_type"],
            eq_method=mcfg["eq_method"],
        )
        mux_results.append((mcfg["label"], res))
        for s, m, d in zip(
            res["snr_db"][::4], res["mux_success_rate"][::4],
            res["data_match_rate"][::4], strict=False,
        ):
            print(f"    {mcfg['label']}  SNR={s:2.0f} dB  MuxOK={m:.3f}  DataMatch={d:.3f}")

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 信令成功率 (多信道对比)
    ax = axes[0]
    markers = ["o-", "s--", "s-", "v-"]
    for (label, res), mk in zip(sig_results, markers, strict=False):
        ax.plot(res["snr_db"], res["signaling_success_rate"], mk,
                label=label, markersize=4)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Phase 9 - Signaling (Channel Comparison)")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)

    # 数据帧 FER (AWGN vs Rayleigh+MMSE, 多载荷)
    ax = axes[1]
    line_styles = ["o-", "s-", "^-", "o--", "s--", "^--"]
    idx = 0
    for label, res in data_results:
        for size, metrics in res["results"].items():
            fer_plot = [max(f, 1e-4) for f in metrics["fer"]]
            ax.semilogy(res["snr_db"], fer_plot, line_styles[idx % len(line_styles)],
                        label=f"{label} {size}B", markersize=3)
            idx += 1
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Frame Error Rate")
    ax.set_title("Phase 9 - Data Frame FER")
    ax.legend(fontsize=6)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim(bottom=1e-4)

    # 复用帧 (不同信道)
    ax = axes[2]
    line_styles = ["o-", "s-", "o--", "s--"]
    for i, (label, res) in enumerate(mux_results):
        ax.plot(res["snr_db"], res["mux_success_rate"],
                line_styles[i * 2], label=f"{label} CRC OK", markersize=4)
        ax.plot(res["snr_db"], res["data_match_rate"],
                line_styles[i * 2 + 1], label=f"{label} Data match", markersize=4)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Phase 9 - Mux Frame (Channel Comparison)")
    ax.legend(fontsize=6)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase9.png", dpi=150)
    print("\nPhase 9 curves saved to ber_phase9.png")


# ═══════════════════════════════════════════════════════════════════════
# Phase 10 — 多链路调度仿真
# ═══════════════════════════════════════════════════════════════════════


def sim_multi_link(
    n_links: int = 3,
    snr_range_db: np.ndarray | None = None,
    n_superframes: int = 20,
    mcs_index: int = 7,
    payload_size: int = 10,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    smf_interval: int = 800,
    seed: int = 42,
) -> dict:
    """多链路调度仿真: 多条链路在超帧内按时间片分时传输。

    使用 ScheduleManager 分配时间资源, 每条链路在各自的事件组窗口内
    发送/接收数据帧, 统计每条链路和总体的 FER。

    :param n_links: 并发链路数。
    :param snr_range_db: 信噪比范围 (dB)。
    :param n_superframes: 每个 SNR 点仿真的超帧数。
    :param mcs_index: 调制编码策略索引。
    :param payload_size: 每帧载荷字节数。
    :param channel_type: 信道类型。
    :param cfo_hz: 载波频率偏移。
    :param eq_method: 均衡方法。
    :param rician_k_db: Rician K 因子。
    :param smf_interval: SMF 间隔 (基础时隙)。
    :param seed: 随机种子。

    :returns: {"snr_db": [...],
         "aggregate_fer": [...],
         "per_link_fer": {link_id: [...]},
         "throughput_ratio": [...]}
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.scheduler import (
        EventTimingParams,
        ScheduleManager,
        ScheduleSlotType,
        SmfScheduleConfig,
        TimeSlice,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    # 配置调度器: 等间隔分配时间片
    sched = ScheduleManager()
    sched.configure_smf(SmfScheduleConfig(smf_interval=smf_interval))

    slice_duration = max(1, smf_interval // (n_links + 1))
    for i in range(n_links):
        link_id = i + 1
        timing = EventTimingParams(
            event_group_period=slice_duration,
            event_period=0,
            event_count=1,
            schedule_slot_type=ScheduleSlotType.T_125US,
            tx_max_offset=3,
            rx_max_offset=3,
        )
        sched.register_link(
            link_id=link_id,
            timing=timing,
            time_slices=[
                TimeSlice(
                    offset=i * slice_duration,
                    duration=slice_duration,
                ),
            ],
        )

    # 验证无时间片冲突
    conflicts = sched.superframe.check_conflicts()

    aggregate_fer_list = []
    per_link_fer: dict[int, list[float]] = {i + 1: [] for i in range(n_links)}
    throughput_list = []

    for snr in snr_range_db:
        link_errors: dict[int, int] = {i + 1: 0 for i in range(n_links)}
        link_total: dict[int, int] = {i + 1: 0 for i in range(n_links)}

        for _ in range(n_superframes):
            for link_id in range(1, n_links + 1):
                schedule = sched.get_event_schedule(link_id)
                if schedule is None:
                    continue

                for _event in schedule:
                    payload = bytes(
                        rng.integers(0, 256, payload_size, dtype=np.uint8)
                    )
                    frame = AsyncDataFrame(segment_type=0, data=payload)
                    mac_bytes = frame.pack()
                    n_mac_bytes = len(mac_bytes)

                    iq = mac_to_iq(mac_bytes, cfg)
                    rx_iq = _channel_impair(
                        iq, float(snr), channel_type, rician_k_db,
                        cfo_hz, eq_method, cfg.sps, rng,
                    )
                    rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

                    link_total[link_id] += 1
                    if not rx.crc_ok:
                        link_errors[link_id] += 1
                    else:
                        try:
                            recovered = AsyncDataFrame.unpack(rx.mac_payload)
                            if recovered.data != payload:
                                link_errors[link_id] += 1
                        except (ValueError, IndexError):
                            link_errors[link_id] += 1

        total_frames = sum(link_total.values())
        total_errors = sum(link_errors.values())
        agg_fer = total_errors / total_frames if total_frames > 0 else 0.0
        aggregate_fer_list.append(agg_fer)
        throughput_list.append(1.0 - agg_fer)

        for link_id in range(1, n_links + 1):
            lt = link_total[link_id]
            le = link_errors[link_id]
            per_link_fer[link_id].append(le / lt if lt > 0 else 0.0)

    return {
        "snr_db": snr_range_db.tolist(),
        "aggregate_fer": aggregate_fer_list,
        "per_link_fer": per_link_fer,
        "throughput_ratio": throughput_list,
        "n_conflicts": len(conflicts),
    }


def sim_access_scheduled_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """接入流程 + 调度器驱动的数据传输仿真。

    模拟完整的接入建链过程, 然后使用接入参数配置调度器,
    在事件组时间窗口内进行数据帧收发。

    :param snr_range_db: 信噪比范围 (dB)。
    :param n_frames: 每个 SNR 点仿真帧数。
    :param mcs_index: MCS 索引。
    :param payload_size: 载荷字节数。
    :param channel_type: 信道类型。
    :param cfo_hz: 载波频率偏移。
    :param eq_method: 均衡方法。
    :param rician_k_db: Rician K 因子。
    :param seed: 随机种子。

    :returns: {"snr_db": [...],
         "fer": [...],
         "access_ok": bool,
         "link_params": dict}
    """
    from nearlink_sdr.mac.access import run_access_procedure
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.link_manager import LinkState
    from nearlink_sdr.mac.scheduler import (
        EventTimingParams,
        ScheduleManager,
        SmfScheduleConfig,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    # 1. 接入流程
    b_mgr, i_mgr = run_access_procedure()
    access_ok = (
        b_mgr.link_manager.state == LinkState.CONNECTED
        and i_mgr.link_manager.state == LinkState.CONNECTED
    )

    if not access_ok:
        return {
            "snr_db": snr_range_db.tolist(),
            "fer": [1.0] * len(snr_range_db),
            "access_ok": False,
            "link_params": {},
        }

    # 2. 使用接入参数配置调度器
    sched = ScheduleManager()
    sched.configure_smf(SmfScheduleConfig(
        smf_interval=b_mgr.smf_period,
        frame_type=b_mgr.smf_frame_type,
    ))
    timing = EventTimingParams(
        event_group_period=b_mgr.access_period,
        event_period=10,
        intra_event_interval=300,
        event_count=2,
        tx_max_offset=3,
        rx_max_offset=3,
    )
    sched.register_link(link_id=b_mgr.access_link_id, timing=timing)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    # 3. 数据传输仿真
    fer_list = []
    link_params = {
        "smf_period": b_mgr.smf_period,
        "access_link_id": b_mgr.access_link_id,
        "access_period": b_mgr.access_period,
    }

    schedule = sched.get_event_schedule(b_mgr.access_link_id)
    events_per_group = len(schedule) if schedule else 1

    for snr in snr_range_db:
        errors = 0

        for _ in range(n_frames):
            payload = bytes(
                rng.integers(0, 256, payload_size, dtype=np.uint8)
            )
            frame = AsyncDataFrame(segment_type=0, data=payload)
            mac_bytes = frame.pack()
            n_mac_bytes = len(mac_bytes)

            iq = mac_to_iq(mac_bytes, cfg)
            rx_iq = _channel_impair(
                iq, float(snr), channel_type, rician_k_db,
                cfo_hz, eq_method, cfg.sps, rng,
            )
            rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

            if not rx.crc_ok:
                errors += 1
            else:
                try:
                    recovered = AsyncDataFrame.unpack(rx.mac_payload)
                    if recovered.data != payload:
                        errors += 1
                except (ValueError, IndexError):
                    errors += 1

        fer_list.append(errors / n_frames)

    return {
        "snr_db": snr_range_db.tolist(),
        "fer": fer_list,
        "access_ok": True,
        "link_params": link_params,
        "events_per_group": events_per_group,
    }


def sim_event_group_timing(
    event_group_period: int = 100,
    event_count: int = 4,
    event_period: int = 25,
    intra_event_interval: int = 300,
    tx_max_offset: int = 3,
    rx_max_offset: int = 3,
) -> dict:
    """事件组时序计算仿真。

    可视化事件组内各事件的 TX/RX 窗口分布和资源利用率。

    :param event_group_period: 事件组周期 (调度时隙)。
    :param event_count: 事件总数。
    :param event_period: 事件周期 (调度时隙)。
    :param intra_event_interval: 事件内间隔 (μs)。
    :param tx_max_offset: TX 最大偏移 (基础时隙)。
    :param rx_max_offset: RX 最大偏移 (基础时隙)。

    :returns: {"events": [{"start": int, "tx_window": (s,e), "rx_window": (s,e)}],
         "total_active_us": int,
         "group_period_us": int,
         "utilization": float}
    """
    from nearlink_sdr.mac.scheduler import EventGroupScheduler, EventTimingParams

    timing = EventTimingParams(
        event_group_period=event_group_period,
        event_period=event_period,
        intra_event_interval=intra_event_interval,
        event_count=event_count,
        tx_max_offset=tx_max_offset,
        rx_max_offset=rx_max_offset,
    )
    scheduler = EventGroupScheduler(timing=timing)
    schedule = scheduler.event_schedule()

    total_active = 0
    for event in schedule:
        tx_s, tx_e = event["tx_window"]
        rx_s, rx_e = event["rx_window"]
        total_active += (tx_e - tx_s) + (rx_e - rx_s)

    group_period_us = timing.event_group_period_us
    utilization = total_active / group_period_us if group_period_us > 0 else 0.0

    return {
        "events": schedule,
        "total_active_us": total_active,
        "group_period_us": group_period_us,
        "utilization": utilization,
    }


def sim_superframe_capacity(
    smf_interval: int = 800,
    slice_duration: int = 50,
    slice_gap: int | None = None,
    max_links: int = 20,
) -> dict:
    """超帧容量分析: 逐步增加链路直到出现时间片冲突。

    :param smf_interval: SMF 间隔 (基础时隙)。
    :param slice_duration: 每条链路时间片持续长度 (调度时隙)。
    :param slice_gap: 相邻链路的偏移间距 (调度时隙, 默认等于 slice_duration)。
        当 gap < slice_duration 时, 链路时间片将发生重叠。
    :param max_links: 最大测试链路数。

    :returns: {"n_links": [...],
         "n_conflicts": [...],
         "max_no_conflict": int,
         "utilization": [...]}
    """
    from nearlink_sdr.mac.scheduler import (
        EventTimingParams,
        ScheduleManager,
        SmfScheduleConfig,
        TimeSlice,
    )

    if slice_gap is None:
        slice_gap = slice_duration

    n_links_list = []
    n_conflicts_list = []
    utilization_list = []
    max_no_conflict = 0

    for n in range(1, max_links + 1):
        sched = ScheduleManager()
        sched.configure_smf(SmfScheduleConfig(smf_interval=smf_interval))

        for i in range(n):
            timing = EventTimingParams(event_group_period=50, event_count=1)
            sched.register_link(
                link_id=i + 1,
                timing=timing,
                time_slices=[
                    TimeSlice(
                        offset=i * slice_gap,
                        duration=slice_duration,
                    ),
                ],
            )

        conflicts = sched.superframe.check_conflicts()
        active_start, active_end = sched.superframe.active_region_us()
        total_us = sched.superframe.duration_us
        util = (active_end - active_start) / total_us if total_us > 0 else 0.0

        n_links_list.append(n)
        n_conflicts_list.append(len(conflicts))
        utilization_list.append(util)

        if len(conflicts) == 0:
            max_no_conflict = n

    return {
        "n_links": n_links_list,
        "n_conflicts": n_conflicts_list,
        "max_no_conflict": max_no_conflict,
        "utilization": utilization_list,
    }


def run_phase10_simulation() -> None:
    """Phase 10: 多链路调度仿真入口。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_range = np.arange(0, 16, 2)

    # ── 10.1 多链路 FER ──
    print("[1/4] Multi-link FER simulation...")
    ml_configs = [
        {"n_links": 1, "label": "1 link"},
        {"n_links": 3, "label": "3 links"},
        {"n_links": 6, "label": "6 links"},
    ]
    ml_results = []
    for mcfg in ml_configs:
        print(f"  {mcfg['label']}...")
        res = sim_multi_link(
            n_links=mcfg["n_links"],
            snr_range_db=snr_range,
            n_superframes=10,
        )
        ml_results.append((mcfg["label"], res))
        for s, f in zip(
            res["snr_db"][::4], res["aggregate_fer"][::4], strict=False,
        ):
            print(f"    {mcfg['label']}  SNR={s:2.0f} dB  FER={f:.4f}")

    # ── 10.2 接入 + 调度链路 ──
    print("[2/4] Access + scheduled link simulation...")
    asl_configs = [
        {"channel_type": "awgn", "eq_method": "none", "label": "AWGN"},
        {"channel_type": "rayleigh", "eq_method": "mmse", "label": "Rayleigh+MMSE"},
    ]
    asl_results = []
    for acfg in asl_configs:
        print(f"  {acfg['label']}...")
        res = sim_access_scheduled_link(
            snr_range_db=snr_range,
            channel_type=acfg["channel_type"],
            eq_method=acfg["eq_method"],
        )
        asl_results.append((acfg["label"], res))
        print(f"    Access OK: {res['access_ok']}")
        for s, f in zip(
            res["snr_db"][::4], res["fer"][::4], strict=False,
        ):
            print(f"    {acfg['label']}  SNR={s:2.0f} dB  FER={f:.4f}")

    # ── 10.3 超帧容量分析 ──
    print("[3/4] Superframe capacity analysis...")
    cap = sim_superframe_capacity(smf_interval=800, slice_duration=50)
    print(f"  Max links without conflict: {cap['max_no_conflict']}")

    # ── 10.4 事件组时序 ──
    print("[4/4] Event group timing analysis...")
    timing = sim_event_group_timing(
        event_group_period=100, event_count=4,
        event_period=25, intra_event_interval=300,
        tx_max_offset=3, rx_max_offset=3,
    )
    print(f"  Events: {len(timing['events'])}")
    print(f"  Active time: {timing['total_active_us']} μs")
    print(f"  Group period: {timing['group_period_us']} μs")
    print(f"  Utilization: {timing['utilization']:.2%}")

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 10.1 多链路 FER
    ax = axes[0, 0]
    markers = ["o-", "s--", "^:"]
    for (label, res), mk in zip(ml_results, markers, strict=False):
        fer_plot = [max(f, 1e-4) for f in res["aggregate_fer"]]
        ax.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=4)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Aggregate FER")
    ax.set_title("Phase 10.1 - Multi-link FER")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim(bottom=1e-4)

    # 10.2 接入 + 调度
    ax = axes[0, 1]
    for (label, res), mk in zip(
        asl_results, ["o-", "s--"], strict=False,
    ):
        fer_plot = [max(f, 1e-4) for f in res["fer"]]
        ax.semilogy(res["snr_db"], fer_plot, mk, label=label, markersize=4)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("FER")
    ax.set_title("Phase 10.2 - Access+Scheduled Link")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim(bottom=1e-4)

    # 10.3 容量
    ax = axes[1, 0]
    ax.plot(cap["n_links"], cap["n_conflicts"], "ro-", label="Conflicts", markersize=4)
    ax2 = ax.twinx()
    ax2.plot(cap["n_links"], cap["utilization"], "b^--",
             label="Utilization", markersize=4)
    ax.set_xlabel("Number of Links")
    ax.set_ylabel("Conflicts", color="r")
    ax2.set_ylabel("Utilization", color="b")
    ax.set_title("Phase 10.3 - Superframe Capacity")
    ax.grid(True, ls="--", alpha=0.5)

    # 10.4 事件组时序
    ax = axes[1, 1]
    for i, event in enumerate(timing["events"]):
        tx_s, tx_e = event["tx_window"]
        rx_s, rx_e = event["rx_window"]
        ax.barh(i, tx_e - tx_s, left=tx_s, height=0.3,
                color="steelblue", label="TX" if i == 0 else "")
        ax.barh(i, rx_e - rx_s, left=rx_s, height=0.3,
                color="coral", label="RX" if i == 0 else "")
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Event Index")
    ax.set_title("Phase 10.4 - Event Group Timing")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase10.png", dpi=150)
    print("\nPhase 10 curves saved to ber_phase10.png")


# ── Phase 11: 接入→配对→加密数据传输端到端仿真 ──


def sim_secure_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """接入 → 配对 → 加密数据传输端到端仿真。

    模拟完整的安全通信建立过程:
    1. 接入建链 (run_access_procedure)
    2. 配对密钥协商 (run_pairing_procedure)
    3. 加密数据帧经 PHY 管道传输

    :returns: {"snr_db", "fer", "access_ok", "pairing_ok", "encrypted": True}
    """
    from cryptography.exceptions import InvalidTag

    from nearlink_sdr.mac.access import run_access_procedure
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.link_manager import LinkState, Role
    from nearlink_sdr.mac.security_manager import (
        FrameCryptoContext,
        run_pairing_procedure,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    # 1. 接入建链
    b_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"
    b_mgr, i_mgr = run_access_procedure(
        broadcaster_addr=b_addr, initiator_addr=t_addr,
    )
    access_ok = (
        b_mgr.link_manager.state == LinkState.CONNECTED
        and i_mgr.link_manager.state == LinkState.CONNECTED
    )

    fail_result = {
        "snr_db": snr_range_db.tolist(),
        "fer": [1.0] * len(snr_range_db),
        "access_ok": False,
        "pairing_ok": False,
        "encrypted": False,
    }
    if not access_ok:
        return fail_result

    # 2. 确定 G/T 角色
    b_role = b_mgr.link_manager.role
    g_addr = b_addr if b_role == Role.G_NODE else t_addr
    t_node_addr = t_addr if b_role == Role.G_NODE else b_addr

    # 3. 配对
    g_pairing, t_pairing = run_pairing_procedure(
        g_address=g_addr, t_address=t_node_addr,
    )
    pairing_ok = g_pairing.is_paired and t_pairing.is_paired
    if not pairing_ok:
        fail_result["access_ok"] = True
        return fail_result

    # 4. 建立加密上下文
    iv_base = rng.bytes(8)
    g_tx_crypto = FrameCryptoContext(
        session_key=g_pairing.session_key,
        iv_base=iv_base,
        direction=0,
        mic_len=4,
        frame_type=2,
        link_id=b_mgr.access_link_id,
    )
    t_rx_crypto = FrameCryptoContext(
        session_key=t_pairing.session_key,
        iv_base=iv_base,
        direction=0,
        mic_len=4,
        frame_type=2,
        link_id=b_mgr.access_link_id,
    )

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    # 5. 加密数据帧传输仿真
    fer_list = []

    for snr in snr_range_db:
        errors = 0

        for _ in range(n_frames):
            payload = bytes(
                rng.integers(0, 256, payload_size, dtype=np.uint8)
            )

            # 加密
            ct, mic = g_tx_crypto.encrypt(payload)
            encrypted_payload = ct + mic

            # MAC 帧封装 + PHY 发射
            frame = AsyncDataFrame(segment_type=0, data=encrypted_payload)
            mac_bytes = frame.pack()
            n_mac_bytes = len(mac_bytes)

            iq = mac_to_iq(mac_bytes, cfg)
            rx_iq = _channel_impair(
                iq, float(snr), channel_type, rician_k_db,
                cfo_hz, eq_method, cfg.sps, rng,
            )
            rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

            if not rx.crc_ok:
                errors += 1
                # CRC 失败时仍需递增 rx 计数器保持同步
                t_rx_crypto._rx_payload_count += 1
                continue

            # 解密
            try:
                recovered_frame = AsyncDataFrame.unpack(rx.mac_payload)
                rx_ct = recovered_frame.data[:-4]
                rx_mic = recovered_frame.data[-4:]
                decrypted = t_rx_crypto.decrypt(rx_ct, rx_mic)
                if decrypted != payload:
                    errors += 1
            except (ValueError, InvalidTag):
                errors += 1

        fer_list.append(errors / n_frames)

    return {
        "snr_db": snr_range_db.tolist(),
        "fer": fer_list,
        "access_ok": True,
        "pairing_ok": True,
        "encrypted": True,
    }


def sim_encrypted_vs_plain(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    channel_type: str = "awgn",
    seed: int = 42,
) -> dict:
    """对比加密与明文传输的误帧率。

    :returns: {"snr_db", "fer_encrypted", "fer_plain"}
    """
    from cryptography.exceptions import InvalidTag

    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.security_manager import (
        FrameCryptoContext,
        run_pairing_procedure,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    # 配对获取密钥
    g_pairing, t_pairing = run_pairing_procedure()
    iv_base = rng.bytes(8)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    fer_encrypted = []
    fer_plain = []

    for snr in snr_range_db:
        enc_errors = 0
        plain_errors = 0

        tx_crypto = FrameCryptoContext(
            session_key=g_pairing.session_key,
            iv_base=iv_base,
            direction=0,
            mic_len=4,
        )
        rx_crypto = FrameCryptoContext(
            session_key=t_pairing.session_key,
            iv_base=iv_base,
            direction=0,
            mic_len=4,
        )

        for _ in range(n_frames):
            payload = bytes(
                rng.integers(0, 256, payload_size, dtype=np.uint8)
            )

            # 加密帧
            ct, mic = tx_crypto.encrypt(payload)
            enc_frame = AsyncDataFrame(segment_type=0, data=ct + mic)
            enc_mac = enc_frame.pack()
            enc_iq = mac_to_iq(enc_mac, cfg)
            enc_rx_iq = _channel_impair(
                enc_iq, float(snr), channel_type, 6.0,
                0.0, "none", cfg.sps, rng,
            )
            enc_rx = iq_to_mac(enc_rx_iq, cfg, len(enc_mac))

            if not enc_rx.crc_ok:
                enc_errors += 1
                rx_crypto._rx_payload_count += 1
            else:
                try:
                    rec = AsyncDataFrame.unpack(enc_rx.mac_payload)
                    dec = rx_crypto.decrypt(rec.data[:-4], rec.data[-4:])
                    if dec != payload:
                        enc_errors += 1
                except (ValueError, InvalidTag):
                    enc_errors += 1

            # 明文帧
            plain_frame = AsyncDataFrame(segment_type=0, data=payload)
            plain_mac = plain_frame.pack()
            plain_iq = mac_to_iq(plain_mac, cfg)
            plain_rx_iq = _channel_impair(
                plain_iq, float(snr), channel_type, 6.0,
                0.0, "none", cfg.sps, rng,
            )
            plain_rx = iq_to_mac(plain_rx_iq, cfg, len(plain_mac))

            if not plain_rx.crc_ok:
                plain_errors += 1
            else:
                try:
                    rec = AsyncDataFrame.unpack(plain_rx.mac_payload)
                    if rec.data != payload:
                        plain_errors += 1
                except ValueError:
                    plain_errors += 1

        fer_encrypted.append(enc_errors / n_frames)
        fer_plain.append(plain_errors / n_frames)

    return {
        "snr_db": snr_range_db.tolist(),
        "fer_encrypted": fer_encrypted,
        "fer_plain": fer_plain,
    }


def sim_pairing_signaling_phy(
    snr_range_db: np.ndarray | None = None,
    n_trials: int = 20,
    channel_type: str = "awgn",
    seed: int = 42,
) -> dict:
    """配对信令经 PHY 管道传输的成功率仿真。

    将配对过程中的每条信令编码为 IQ, 经信道传输后解码,
    统计信令传输成功率。

    :returns: {"snr_db", "signaling_success_rate"}
    """
    from nearlink_sdr.mac.security_manager import PairingManager
    from nearlink_sdr.phy.mac_interface import (
        iq_to_signaling,
        signaling_to_iq,
    )
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 20, 2)

    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=0,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    success_rates = []

    for snr in snr_range_db:
        total_msgs = 0
        successes = 0

        for _ in range(n_trials):
            # 生成一次完整配对的信令序列
            g = PairingManager(is_g_node=True)
            t = PairingManager(is_g_node=False)
            g_msgs = g.start_pairing()
            t.start_pairing()

            pending = [("g", m) for m in g_msgs]
            max_rounds = 20

            for _ in range(max_rounds):
                if not pending:
                    break
                next_pending = []
                for source, msg in pending:
                    total_msgs += 1
                    # 信令经 PHY 管道传输
                    try:
                        iq = signaling_to_iq(msg, cfg)
                        # ControlFrame: 2B header + 1B length + payload
                        n_bytes = msg.BYTE_LENGTH + 3
                        rx_iq = _channel_impair(
                            iq, float(snr), channel_type, 6.0,
                            0.0, "none", cfg.sps, rng,
                        )
                        rx_msg, _ok = iq_to_signaling(rx_iq, cfg, n_bytes)
                        if rx_msg is not None:
                            # 将恢复的信令传给对端
                            successes += 1
                            if source == "g":
                                resps = t.process_message(rx_msg)
                                next_pending.extend(
                                    ("t", r) for r in resps
                                )
                            else:
                                resps = g.process_message(rx_msg)
                                next_pending.extend(
                                    ("g", r) for r in resps
                                )
                    except (ValueError, RuntimeError):
                        pass

                pending = next_pending
                if g.is_paired and t.is_paired:
                    break

        rate = successes / total_msgs if total_msgs > 0 else 0
        success_rates.append(rate)

    return {
        "snr_db": snr_range_db.tolist(),
        "signaling_success_rate": success_rates,
    }


def run_phase11_simulation() -> None:
    """Phase 11: 安全通信端到端仿真可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 11 - Secure Link E2E Simulation", fontsize=14)
    snr = np.arange(0, 16, 2)

    # 11.1 加密数据链路 FER
    ax = axes[0, 0]
    result = sim_secure_link(snr_range_db=snr, n_frames=30)
    ax.semilogy(result["snr_db"], result["fer"], "bo-", label="Encrypted FER")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("Phase 11.1 - Secure Link FER")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # 11.2 加密 vs 明文 FER 对比
    ax = axes[0, 1]
    cmp = sim_encrypted_vs_plain(snr_range_db=snr, n_frames=30)
    ax.semilogy(cmp["snr_db"], cmp["fer_encrypted"], "rs-", label="Encrypted")
    ax.semilogy(cmp["snr_db"], cmp["fer_plain"], "b^--", label="Plain")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("Phase 11.2 - Encrypted vs Plain FER")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # 11.3 配对信令 PHY 传输成功率
    ax = axes[1, 0]
    sig = sim_pairing_signaling_phy(snr_range_db=snr, n_trials=5)
    ax.plot(sig["snr_db"], sig["signaling_success_rate"], "go-",
            label="Signaling Success Rate")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Phase 11.3 - Pairing Signaling via PHY")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # 11.4 多信道类型加密链路
    ax = axes[1, 1]
    for ch_type in ("awgn", "rayleigh"):
        r = sim_secure_link(
            snr_range_db=snr, n_frames=20, channel_type=ch_type,
        )
        ax.semilogy(r["snr_db"], r["fer"], "o-", label=ch_type.upper())
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("Phase 11.4 - Encrypted Link: AWGN vs Rayleigh")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase11.png", dpi=150)
    print("\nPhase 11 curves saved to ber_phase11.png")


# ========================================================================
# Phase 12: AMC 自适应调制编码 + HARQ 重传仿真
# ========================================================================


def sim_amc_throughput(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    payload_size: int = 10,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    mcs_indices: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """各 MCS 级别的吞吐量与误帧率扫描, 含 AMC 包络线。

    遍历所有 MCS 索引 (0-12), 对每个 SNR 点计算 FER 和有效吞吐量。
    有效吞吐量 = (1 - FER) * spectral_efficiency (bit/symbol)。
    AMC 策略: 在每个 SNR 点选择吞吐量最高的 MCS。

    :returns: {
            "snr_db": [...],
            "mcs_fer": {mcs_idx: [fer_per_snr, ...]},
            "mcs_throughput": {mcs_idx: [throughput_per_snr, ...]},
            "amc_throughput": [...],  # AMC 包络吞吐量
            "amc_mcs": [...],         # AMC 选择的 MCS 索引
        }
    """
    from nearlink_sdr.common.mcs import MCS_TABLE
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(-2, 22, 1)

    if mcs_indices is None:
        mcs_indices = [e.index for e in MCS_TABLE]

    mcs_fer: dict[int, list[float]] = {}
    mcs_throughput: dict[int, list[float]] = {}

    for mcs_idx in mcs_indices:
        mcs_entry = MCS_TABLE[mcs_idx]
        fer_list: list[float] = []
        tp_list: list[float] = []

        cfg = TxConfig(
            frame_type=2,
            mcs_index=mcs_idx,
            pid=0x123456,
            whitening_seed=0x52,
            crc_seed=0x555555,
            crc_len=24,
            ctrl_bits_len=28,
            pilot_interval=8,
        )

        for snr in snr_range_db:
            frame_errors = 0
            frame_rng = np.random.default_rng(
                seed + mcs_idx * 1000 + int(snr * 10)
            )

            for _ in range(n_frames):
                mac_payload = bytes(
                    frame_rng.integers(0, 256, payload_size, dtype=np.uint8)
                )
                n_mac_bytes = len(mac_payload)

                iq = mac_to_iq(mac_payload, cfg)
                rx_iq = _channel_impair(
                    iq, float(snr), channel_type, rician_k_db,
                    cfo_hz, eq_method, cfg.sps, frame_rng,
                )
                rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

                if not rx.crc_ok or rx.mac_payload != mac_payload:
                    frame_errors += 1

            fer = frame_errors / n_frames
            fer_list.append(fer)
            tp_list.append((1.0 - fer) * mcs_entry.spectral_efficiency)

        mcs_fer[mcs_idx] = fer_list
        mcs_throughput[mcs_idx] = tp_list

    # AMC 包络: 每个 SNR 点选择吞吐量最大的 MCS
    n_snr = len(snr_range_db)
    amc_throughput = []
    amc_mcs = []
    for i in range(n_snr):
        best_tp = -1.0
        best_mcs = 0
        for mcs_idx in mcs_indices:
            tp = mcs_throughput[mcs_idx][i]
            if tp > best_tp:
                best_tp = tp
                best_mcs = mcs_idx
        amc_throughput.append(best_tp)
        amc_mcs.append(best_mcs)

    return {
        "snr_db": snr_range_db.tolist(),
        "mcs_fer": mcs_fer,
        "mcs_throughput": mcs_throughput,
        "amc_throughput": amc_throughput,
        "amc_mcs": amc_mcs,
    }


def sim_harq_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    max_retries: int = 3,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """HARQ 重传链路仿真 (Chase combining 简化模型)。

    每帧传输失败时进行重传, 最多重传 max_retries 次。
    使用 B1 控制信息中的 harq_feedback 和 packet_sn 字段。
    标准 6.10.5 规定 MCS=15 表示重传帧。

    :returns: {
            "snr_db": [...],
            "fer_no_harq": [...],     # 无重传 FER
            "fer_harq": [...],        # HARQ FER (超过最大重传仍失败)
            "avg_transmissions": [...], # 平均每帧传输次数
            "throughput_no_harq": [...],  # 无重传吞吐量 (归一化)
            "throughput_harq": [...],     # HARQ 吞吐量 (归一化)
        }
    """
    from nearlink_sdr.common.mcs import get_mcs
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    mcs_entry = get_mcs(mcs_index)
    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    fer_no_harq_list = []
    fer_harq_list = []
    avg_tx_list = []
    tp_no_harq_list = []
    tp_harq_list = []

    for snr in snr_range_db:
        no_harq_errors = 0
        harq_errors = 0
        total_transmissions = 0

        for _ in range(n_frames):
            mac_payload = bytes(
                rng.integers(0, 256, payload_size, dtype=np.uint8)
            )
            n_mac_bytes = len(mac_payload)

            # 首次传输
            iq = mac_to_iq(mac_payload, cfg)
            rx_iq = _channel_impair(
                iq, float(snr), channel_type, rician_k_db,
                cfo_hz, eq_method, cfg.sps, rng,
            )
            rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)

            first_ok = rx.crc_ok and rx.mac_payload == mac_payload
            if not first_ok:
                no_harq_errors += 1

            # HARQ 重传循环
            attempts = 1
            success = first_ok
            while not success and attempts <= max_retries:
                attempts += 1
                iq_retx = mac_to_iq(mac_payload, cfg)
                rx_iq_retx = _channel_impair(
                    iq_retx, float(snr), channel_type, rician_k_db,
                    cfo_hz, eq_method, cfg.sps, rng,
                )
                rx_retx = iq_to_mac(rx_iq_retx, cfg, n_mac_bytes)
                success = rx_retx.crc_ok and rx_retx.mac_payload == mac_payload

            if not success:
                harq_errors += 1
            total_transmissions += attempts

        fer_no_harq = no_harq_errors / n_frames
        fer_harq = harq_errors / n_frames
        avg_tx = total_transmissions / n_frames

        fer_no_harq_list.append(fer_no_harq)
        fer_harq_list.append(fer_harq)
        avg_tx_list.append(avg_tx)

        se = mcs_entry.spectral_efficiency
        tp_no_harq_list.append((1.0 - fer_no_harq) * se)
        tp_harq_list.append((1.0 - fer_harq) * se / avg_tx)

    return {
        "snr_db": snr_range_db.tolist(),
        "fer_no_harq": fer_no_harq_list,
        "fer_harq": fer_harq_list,
        "avg_transmissions": avg_tx_list,
        "throughput_no_harq": tp_no_harq_list,
        "throughput_harq": tp_harq_list,
    }


def sim_hopping_multipath_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    n_hop_channels: int = 8,
    seed: int = 42,
) -> dict:
    """跳频 + 多径信道链路仿真。

    每帧在不同频率信道上传输, 模拟跳频对抗频率选择性衰落的效果。
    对比固定信道 (Rayleigh) 与跳频 (不同信道独立衰落) 的 FER。

    :returns: {
            "snr_db": [...],
            "fer_fixed": [...],     # 固定信道 FER
            "fer_hopping": [...],   # 跳频 FER
        }
    """
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 20, 2)

    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    fer_fixed_list = []
    fer_hopping_list = []

    for snr in snr_range_db:
        fixed_errors = 0
        hopping_errors = 0

        for i in range(n_frames):
            mac_payload = bytes(
                rng.integers(0, 256, payload_size, dtype=np.uint8)
            )
            n_mac_bytes = len(mac_payload)

            iq = mac_to_iq(mac_payload, cfg)

            # 固定信道: 使用同一衰落种子 (深衰落持续)
            fixed_rng = np.random.default_rng(seed + int(snr * 10))
            rx_iq_fixed = _channel_impair(
                iq, float(snr), "rayleigh", 6.0,
                0.0, "none", cfg.sps, fixed_rng,
            )
            rx_fixed = iq_to_mac(rx_iq_fixed, cfg, n_mac_bytes)
            if not rx_fixed.crc_ok or rx_fixed.mac_payload != mac_payload:
                fixed_errors += 1

            # 跳频信道: 每帧独立衰落种子 (模拟频率分集)
            hop_seed = seed + i * 1000 + int(snr * 10)
            hop_rng = np.random.default_rng(hop_seed)
            rx_iq_hop = _channel_impair(
                iq, float(snr), "rayleigh", 6.0,
                0.0, "none", cfg.sps, hop_rng,
            )
            rx_hop = iq_to_mac(rx_iq_hop, cfg, n_mac_bytes)
            if not rx_hop.crc_ok or rx_hop.mac_payload != mac_payload:
                hopping_errors += 1

        fer_fixed_list.append(fixed_errors / n_frames)
        fer_hopping_list.append(hopping_errors / n_frames)

    return {
        "snr_db": snr_range_db.tolist(),
        "fer_fixed": fer_fixed_list,
        "fer_hopping": fer_hopping_list,
    }


def run_phase12_simulation() -> None:
    """Phase 12: AMC 吞吐量 + HARQ 重传 + 跳频多径仿真可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 12: AMC / HARQ / Hopping Simulation", fontsize=14)

    # --- 12.1 各 MCS FER ---
    snr_amc = np.arange(-2, 22, 1)
    amc = sim_amc_throughput(snr_range_db=snr_amc, n_frames=30)

    ax = axes[0, 0]
    for mcs_idx, fer in amc["mcs_fer"].items():
        ax.semilogy(amc["snr_db"], np.array(fer) + 1e-6, label=f"MCS {mcs_idx}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("12.1 - FER per MCS")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)

    # --- 12.2 各 MCS 吞吐量 + AMC 包络 ---
    ax = axes[0, 1]
    for mcs_idx, tp in amc["mcs_throughput"].items():
        ax.plot(amc["snr_db"], tp, "--", alpha=0.5, label=f"MCS {mcs_idx}")
    ax.plot(amc["snr_db"], amc["amc_throughput"], "k-", lw=2, label="AMC")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Throughput (bit/symbol)")
    ax.set_title("12.2 - AMC Throughput Envelope")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)

    # --- 12.3 AMC 选择的 MCS ---
    ax = axes[0, 2]
    ax.step(amc["snr_db"], amc["amc_mcs"], where="mid")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MCS Index")
    ax.set_title("12.3 - AMC MCS Selection")
    ax.set_yticks(range(13))
    ax.grid(True, ls="--", alpha=0.5)

    # --- 12.4 HARQ FER 对比 ---
    snr_harq = np.arange(0, 16, 1)
    harq = sim_harq_link(snr_range_db=snr_harq, n_frames=50)

    ax = axes[1, 0]
    ax.semilogy(harq["snr_db"], np.array(harq["fer_no_harq"]) + 1e-6,
                "o-", label="No HARQ")
    ax.semilogy(harq["snr_db"], np.array(harq["fer_harq"]) + 1e-6,
                "s-", label="HARQ (max 3)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("12.4 - HARQ FER Improvement")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # --- 12.5 HARQ 吞吐量与平均传输次数 ---
    ax = axes[1, 1]
    ax.plot(harq["snr_db"], harq["throughput_no_harq"], "o-", label="No HARQ")
    ax.plot(harq["snr_db"], harq["throughput_harq"], "s-", label="HARQ")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Throughput (bit/symbol)")
    ax.set_title("12.5 - HARQ Throughput")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(harq["snr_db"], harq["avg_transmissions"], "^--",
             color="gray", alpha=0.6, label="Avg TX")
    ax2.set_ylabel("Avg Transmissions")
    ax2.legend(loc="center right")

    # --- 12.6 跳频 vs 固定信道 ---
    snr_hop = np.arange(0, 20, 1)
    hop = sim_hopping_multipath_link(snr_range_db=snr_hop, n_frames=50)

    ax = axes[1, 2]
    ax.semilogy(hop["snr_db"], np.array(hop["fer_fixed"]) + 1e-6,
                "o-", label="Fixed Channel")
    ax.semilogy(hop["snr_db"], np.array(hop["fer_hopping"]) + 1e-6,
                "s-", label="Freq Hopping")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("12.6 - Hopping vs Fixed (Rayleigh)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase12.png", dpi=150)
    print("\nPhase 12 curves saved to ber_phase12.png")


# ===================================================================
# Phase 13: QoS 驱动的 ARQ/HARQ + 流控 + AMC 仿真
# ===================================================================


def sim_qos_arq_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    max_retries: int = 3,
    channel_type: str = "awgn",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """QoS 管理器驱动的 ARQ 重传链路仿真。

    使用 QosLink 封装实现自动 ARQ 重传, 对比无 QoS 管理时的 FER。
    验证 QosManager 的序列号管理、重传决策和链路质量跟踪。

    :returns: {
            "snr_db": [...],
            "fer_no_arq": [...],       # 无重传 FER
            "fer_qos_arq": [...],      # QoS ARQ FER
            "avg_transmissions": [...], # 平均每帧传输次数
            "throughput_no_arq": [...],
            "throughput_qos_arq": [...],
        }
    """
    from nearlink_sdr.common.mcs import get_mcs
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.qos import ArqState, LinkType, QosManager, TxDecision
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    mcs_entry = get_mcs(mcs_index)
    rng = np.random.default_rng(seed)

    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    fer_no_arq_list = []
    fer_qos_arq_list = []
    avg_tx_list = []
    tp_no_arq = []
    tp_qos_arq = []

    for snr in snr_range_db:
        no_arq_errors = 0
        qos_arq_errors = 0
        total_transmissions = 0

        for _ in range(n_frames):
            mac_payload = bytes(
                rng.integers(0, 256, payload_size, dtype=np.uint8)
            )
            frame = AsyncDataFrame(segment_type=0, data=mac_payload)
            mac_bytes = frame.pack()
            n_mac_bytes = len(mac_bytes)

            # 无 ARQ 单次传输
            iq = mac_to_iq(mac_bytes, cfg)
            rx_iq = _channel_impair(
                iq, float(snr), channel_type, rician_k_db,
                0.0, "none", cfg.sps, rng,
            )
            rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)
            first_ok = rx.crc_ok and rx.mac_payload == mac_bytes
            if not first_ok:
                no_arq_errors += 1

            # QoS ARQ 重传
            qos = QosManager(
                arq=ArqState(frame_type=2, link_type=LinkType.SYNC,
                             max_retransmit=max_retries),
            )
            qos.submit_data(mac_payload)
            _decision, _item = qos.prepare_tx()
            attempts = 1
            success = first_ok

            while not success and attempts <= max_retries:
                feedback = qos.on_tx_feedback(crc_ok=False)
                if feedback == TxDecision.NEW_DATA:
                    break

                attempts += 1
                iq_retx = mac_to_iq(mac_bytes, cfg)
                rx_iq_retx = _channel_impair(
                    iq_retx, float(snr), channel_type, rician_k_db,
                    0.0, "none", cfg.sps, rng,
                )
                rx_retx = iq_to_mac(rx_iq_retx, cfg, n_mac_bytes)
                success = rx_retx.crc_ok and rx_retx.mac_payload == mac_bytes

            if success:
                qos.on_tx_feedback(crc_ok=True)
            else:
                qos_arq_errors += 1

            total_transmissions += attempts

        fer_no_arq = no_arq_errors / n_frames
        fer_qos_arq = qos_arq_errors / n_frames
        avg_tx = total_transmissions / n_frames

        fer_no_arq_list.append(fer_no_arq)
        fer_qos_arq_list.append(fer_qos_arq)
        avg_tx_list.append(avg_tx)

        se = mcs_entry.spectral_efficiency
        tp_no_arq.append((1.0 - fer_no_arq) * se)
        tp_qos_arq.append((1.0 - fer_qos_arq) * se / max(avg_tx, 1.0))

    return {
        "snr_db": snr_range_db.tolist(),
        "fer_no_arq": fer_no_arq_list,
        "fer_qos_arq": fer_qos_arq_list,
        "avg_transmissions": avg_tx_list,
        "throughput_no_arq": tp_no_arq,
        "throughput_qos_arq": tp_qos_arq,
    }


def sim_qos_amc_adaptive(
    snr_sequence: np.ndarray | None = None,
    n_frames_per_snr: int = 20,
    payload_size: int = 10,
    channel_type: str = "awgn",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """QoS 链路质量跟踪驱动的 AMC 自适应仿真。

    模拟 SNR 随时间变化的场景, QoS 管理器根据 FER
    反馈自动调整 MCS, 验证 AMC 跟踪性能。

    :returns: {
            "snr_trace": [...],
            "mcs_trace": [...],
            "fer_trace": [...],
            "throughput_trace": [...],
        }
    """
    from nearlink_sdr.common.mcs import get_mcs
    from nearlink_sdr.mac.qos import LinkQualityTracker
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if snr_sequence is None:
        snr_sequence = np.concatenate([
            np.full(n_frames_per_snr, 5),
            np.full(n_frames_per_snr, 12),
            np.full(n_frames_per_snr, 3),
            np.full(n_frames_per_snr, 18),
            np.full(n_frames_per_snr, 8),
        ])

    rng = np.random.default_rng(seed)
    tracker = LinkQualityTracker(window_size=8)
    tracker._current_mcs = 5

    snr_trace = []
    mcs_trace = []
    fer_trace = []
    tp_trace = []

    for snr in snr_sequence:
        mcs_idx = tracker.current_mcs
        mcs_entry = get_mcs(mcs_idx)

        cfg = TxConfig(
            frame_type=2,
            mcs_index=mcs_idx,
            pid=0x123456,
            whitening_seed=0x52,
            crc_seed=0x555555,
            crc_len=24,
            ctrl_bits_len=28,
            pilot_interval=8,
        )

        mac_payload = bytes(
            rng.integers(0, 256, payload_size, dtype=np.uint8)
        )
        from nearlink_sdr.mac.frame import AsyncDataFrame
        frame = AsyncDataFrame(segment_type=0, data=mac_payload)
        mac_bytes = frame.pack()
        n_mac_bytes = len(mac_bytes)

        iq = mac_to_iq(mac_bytes, cfg)
        rx_iq = _channel_impair(
            iq, float(snr), channel_type, rician_k_db,
            0.0, "none", cfg.sps, rng,
        )
        rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)
        crc_ok = rx.crc_ok and rx.mac_payload == mac_bytes

        tracker.record(crc_ok)
        tracker.apply_suggestion()

        snr_trace.append(float(snr))
        mcs_trace.append(mcs_idx)
        fer_trace.append(tracker.fer)
        tp_trace.append(
            (1.0 if crc_ok else 0.0) * mcs_entry.spectral_efficiency
        )

    return {
        "snr_trace": snr_trace,
        "mcs_trace": mcs_trace,
        "fer_trace": fer_trace,
        "throughput_trace": tp_trace,
    }


def sim_qos_flow_control(
    n_frames: int = 100,
    burst_size: int = 10,
    snr_db: float = 15.0,
    mcs_index: int = 7,
    payload_size: int = 10,
    high_watermark: int = 8,
    low_watermark: int = 2,
    channel_type: str = "awgn",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """QoS 流控机制仿真。

    模拟突发数据到达场景, 验证流控背压机制对缓冲区占用和丢包的影响。

    :returns: {
            "frame_idx": [...],
            "buffer_occupancy": [...],
            "flow_ctrl_bit": [...],
            "paused": [...],
            "tx_success": [...],
        }
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.qos import FlowController, Priority, TxQueue, TxQueueItem
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    rng = np.random.default_rng(seed)
    cfg = TxConfig(
        frame_type=2,
        mcs_index=mcs_index,
        pid=0x123456,
        whitening_seed=0x52,
        crc_seed=0x555555,
        crc_len=24,
        ctrl_bits_len=28,
        pilot_interval=8,
    )

    flow = FlowController(
        buffer_high_watermark=high_watermark,
        buffer_low_watermark=low_watermark,
    )
    tx_queue = TxQueue(max_size=32)

    frame_idx = []
    buffer_occ = []
    fc_bit = []
    paused = []
    tx_success_list = []

    for i in range(n_frames):
        # 每隔 burst_size 帧产生一批数据
        if i % burst_size == 0:
            n_burst = rng.integers(3, burst_size + 1)
            for _ in range(n_burst):
                payload = bytes(
                    rng.integers(0, 256, payload_size, dtype=np.uint8)
                )
                item = TxQueueItem(priority=Priority.NORMAL, data=payload)
                if tx_queue.push(item):
                    flow.enqueue()

        # 发送一帧 (如果有数据且未暂停)
        tx_ok = False
        if not flow.is_paused and not tx_queue.is_empty:
            item = tx_queue.pop()
            flow.dequeue()

            frame = AsyncDataFrame(segment_type=0, data=item.data)
            mac_bytes = frame.pack()
            n_mac_bytes = len(mac_bytes)

            iq = mac_to_iq(mac_bytes, cfg)
            rx_iq = _channel_impair(
                iq, snr_db, channel_type, rician_k_db,
                0.0, "none", cfg.sps, rng,
            )
            rx = iq_to_mac(rx_iq, cfg, n_mac_bytes)
            tx_ok = rx.crc_ok

        frame_idx.append(i)
        buffer_occ.append(flow.buffer_count)
        fc_bit.append(flow.flow_ctrl_bit)
        paused.append(flow.is_paused)
        tx_success_list.append(tx_ok)

    return {
        "frame_idx": frame_idx,
        "buffer_occupancy": buffer_occ,
        "flow_ctrl_bit": fc_bit,
        "paused": paused,
        "tx_success": tx_success_list,
    }


def run_phase13_simulation() -> None:
    """Phase 13: QoS 驱动的 ARQ + AMC 自适应 + 流控仿真可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 13: QoS ARQ / AMC Adaptive / Flow Control", fontsize=14)

    # --- 13.1 QoS ARQ FER 对比 ---
    snr_arq = np.arange(0, 16, 1)
    arq = sim_qos_arq_link(snr_range_db=snr_arq, n_frames=30)

    ax = axes[0, 0]
    ax.semilogy(arq["snr_db"], np.array(arq["fer_no_arq"]) + 1e-6,
                "o-", label="No ARQ")
    ax.semilogy(arq["snr_db"], np.array(arq["fer_qos_arq"]) + 1e-6,
                "s-", label="QoS ARQ")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("13.1 - QoS ARQ FER")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    # --- 13.2 QoS ARQ 吞吐量 ---
    ax = axes[0, 1]
    ax.plot(arq["snr_db"], arq["throughput_no_arq"], "o-", label="No ARQ")
    ax.plot(arq["snr_db"], arq["throughput_qos_arq"], "s-", label="QoS ARQ")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Throughput (bit/symbol)")
    ax.set_title("13.2 - QoS ARQ Throughput")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(arq["snr_db"], arq["avg_transmissions"], "^--",
             color="gray", alpha=0.6, label="Avg TX")
    ax2.set_ylabel("Avg Transmissions")
    ax2.legend(loc="center right")

    # --- 13.3 AMC 自适应 MCS 跟踪 ---
    amc = sim_qos_amc_adaptive(n_frames_per_snr=20)

    ax = axes[0, 2]
    frames = range(len(amc["snr_trace"]))
    ax.plot(list(frames), amc["snr_trace"], "-", alpha=0.5, label="SNR")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("13.3 - AMC Adaptive MCS Tracking")
    ax.legend(loc="upper left")
    ax.grid(True, ls="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.step(list(frames), amc["mcs_trace"], "r-", where="mid", label="MCS")
    ax2.set_ylabel("MCS Index")
    ax2.set_yticks(range(13))
    ax2.legend(loc="upper right")

    # --- 13.4 AMC 帧吞吐量 ---
    ax = axes[1, 0]
    ax.plot(list(frames), amc["throughput_trace"], "-", alpha=0.7)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Throughput (bit/symbol)")
    ax.set_title("13.4 - AMC Frame Throughput")
    ax.grid(True, ls="--", alpha=0.5)

    # --- 13.5 流控缓冲区占用 ---
    fc = sim_qos_flow_control(n_frames=100)

    ax = axes[1, 1]
    ax.bar(fc["frame_idx"], fc["buffer_occupancy"], width=1.0, alpha=0.7,
           label="Buffer")
    ax.axhline(y=8, color="r", ls="--", alpha=0.5, label="High WM")
    ax.axhline(y=2, color="g", ls="--", alpha=0.5, label="Low WM")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Buffer Occupancy")
    ax.set_title("13.5 - Flow Control Buffer")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    # --- 13.6 流控暂停与传输成功 ---
    ax = axes[1, 2]
    ax.fill_between(fc["frame_idx"], [int(p) for p in fc["paused"]],
                     alpha=0.3, color="red", label="Paused")
    ax.step(fc["frame_idx"], fc["flow_ctrl_bit"], "b-", where="mid",
            label="flow_ctrl", alpha=0.7)
    success_frames = [i for i, s in zip(fc["frame_idx"], fc["tx_success"], strict=False) if s]
    ax.scatter(success_frames, [1.1] * len(success_frames), c="green",
               s=5, label="TX OK")
    ax.set_xlabel("Frame Index")
    ax.set_title("13.6 - Flow Control & TX Status")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase13.png", dpi=150)
    print("\nPhase 13 curves saved to ber_phase13.png")


# ===================================================================
# Phase 14: 双节点端到端仿真
# ===================================================================


def sim_dual_node_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 100,
    channel_type: str = "awgn",
    cfo_hz: float = 0.0,
    eq_method: str = "none",
    rician_k_db: float = 6.0,
    seed: int = 42,
) -> dict:
    """双 SleNode 实体数据交换仿真。

    创建 G 节点与 T 节点, 完成广播→接入→数据交换全流程。
    G 节点发送随机数据, T 节点接收并统计 FER 与字节误码率。

    :returns: {
            "snr_db": [...],
            "fer": [...],
            "byte_ber": [...],
            "tx_count": int,
            "rx_count": int,
        }
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.node import NodeConfig, NodeRole, SleNode

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    g_node = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=mcs_index, max_retransmit=0,
    ))
    t_node = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=mcs_index, max_retransmit=0,
    ))

    # 建连
    g_node.start_advertising()
    t_node.start_scanning()
    t_node.connect(g_addr)
    from nearlink_sdr.mac.link_manager import Role
    g_node.accept_connection(t_addr, Role.G_NODE)

    fer_list, ber_list = [], []

    for snr in snr_range_db:
        frame_errors, byte_errors, total_bytes = 0, 0, 0

        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            g_node.send(payload)
            tx = g_node.transmit()

            if tx.iq is None:
                frame_errors += 1
                total_bytes += payload_size
                # 清除 ARQ 重传锁定, 使下一帧可正常发送
                g_node._qos.arq.on_ack_received()
                continue

            rx_iq = _channel_impair(
                tx.iq, float(snr), channel_type, rician_k_db,
                cfo_hz, eq_method, g_node._tx_config.sps, rng,
            )

            frame = AsyncDataFrame(segment_type=0, data=payload)
            n_mac = len(frame.pack())
            rx = t_node.receive(rx_iq, n_mac)
            g_node.process_feedback(rx.success)
            if not rx.success:
                # 清除 ARQ 重传锁定, FER 仿真中每帧独立
                g_node._qos.arq.on_ack_received()

            total_bytes += payload_size
            if not rx.success:
                frame_errors += 1
            # 逐字节比对: 即使 CRC 失败, 部分字节可能仍然正确
            if rx.data is not None:
                byte_errors += sum(
                    a != b for a, b in zip(payload, rx.data)
                )
            else:
                byte_errors += payload_size

        fer_list.append(frame_errors / n_frames)
        ber_list.append(byte_errors / max(total_bytes, 1))

    return {
        "snr_db": snr_range_db.tolist(),
        "fer": fer_list,
        "byte_ber": ber_list,
        "tx_count": g_node.stats["tx_count"],
        "rx_count": t_node.stats["rx_count"],
    }


def sim_dual_node_secure_link(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    channel_type: str = "awgn",
    seed: int = 42,
) -> dict:
    """双 SleNode 安全通信仿真 (配对 + 加密)。

    在 sim_dual_node_link 基础上增加配对与加密流程:
    1. 广播→接入→连接
    2. 配对 (ECDH 密钥协商)
    3. 加密数据帧传输

    :returns: {
            "snr_db": [...],
            "fer_plain": [...],
            "fer_encrypted": [...],
            "pairing_ok": bool,
        }
    """
    from nearlink_sdr.mac.link_manager import Role
    from nearlink_sdr.mac.security_manager import FrameCryptoContext
    from nearlink_sdr.node import NodeConfig, NodeRole, SleNode

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    # --- 明文链路 ---
    g_plain = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=mcs_index,
    ))
    t_plain = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=mcs_index,
    ))
    g_plain.start_advertising()
    g_plain.accept_connection(t_addr, Role.G_NODE)
    t_plain.start_scanning()
    t_plain.connect(g_addr)

    fer_plain = _run_dual_frames(
        g_plain, t_plain, snr_range_db, n_frames, payload_size, rng, "awgn",
    )

    # --- 加密链路 ---
    g_enc = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=mcs_index, enable_encryption=True,
    ))
    t_enc = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=mcs_index, enable_encryption=True,
    ))
    g_enc.start_advertising()
    g_enc.accept_connection(t_addr, Role.G_NODE)
    t_enc.start_scanning()
    t_enc.connect(g_addr)

    # 配对: G 节点发起
    g_msgs = g_enc.start_pairing(t_addr)
    t_msgs = []
    for msg in g_msgs:
        t_msgs = t_enc.process_pairing_message(msg)
    # T 节点回复
    g_msgs = []
    for msg in t_msgs:
        g_msgs = g_enc.process_pairing_message(msg)
    # 继续交换直到双方完成
    while g_msgs or t_msgs:
        new_t = []
        for msg in g_msgs:
            new_t.extend(t_enc.process_pairing_message(msg))
        t_msgs = new_t
        new_g = []
        for msg in t_msgs:
            new_g.extend(g_enc.process_pairing_message(msg))
        g_msgs = new_g

    pairing_ok = g_enc.stats["paired"] and t_enc.stats["paired"]

    if pairing_ok:
        # 手动建立匹配的加密上下文 (真实场景由配对过程自动完成)
        iv_base = rng.bytes(8)
        g_enc._crypto = FrameCryptoContext(
            session_key=g_enc._pairing.session_key,
            iv_base=iv_base, direction=0, mic_len=4,
            frame_type=2, link_id=0,
        )
        t_enc._crypto = FrameCryptoContext(
            session_key=t_enc._pairing.session_key if t_enc._pairing else b"\x00" * 16,
            iv_base=iv_base, direction=0, mic_len=4,
            frame_type=2, link_id=0,
        )

    fer_enc = _run_dual_frames(
        g_enc, t_enc, snr_range_db, n_frames, payload_size,
        np.random.default_rng(seed), channel_type,
    )

    return {
        "snr_db": snr_range_db.tolist(),
        "fer_plain": fer_plain,
        "fer_encrypted": fer_enc,
        "pairing_ok": pairing_ok,
    }


def sim_dual_node_mcs_adapt(
    snr_db: float = 8.0,
    n_frames: int = 100,
    payload_size: int = 10,
    initial_mcs: int = 7,
    channel_type: str = "awgn",
    seed: int = 42,
) -> dict:
    """双节点 MCS 自适应仿真。

    固定 SNR 下连续帧传输, 跟踪 G 节点的 MCS 自适应过程和吞吐量变化。

    :returns: {
            "frame_idx": [...],
            "mcs_history": [...],
            "fer_history": [...],
            "success_history": [...],
        }
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.link_manager import Role
    from nearlink_sdr.node import NodeConfig, NodeRole, SleNode

    rng = np.random.default_rng(seed)

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    g_node = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=initial_mcs,
    ))
    t_node = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=initial_mcs,
    ))
    g_node.start_advertising()
    g_node.accept_connection(t_addr, Role.G_NODE)
    t_node.start_scanning()
    t_node.connect(g_addr)

    frame_idx, mcs_hist, fer_hist, success_hist = [], [], [], []

    for i in range(n_frames):
        payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
        g_node.send(payload)
        tx = g_node.transmit()

        success = False
        if tx.iq is not None:
            rx_iq = _channel_impair(
                tx.iq, snr_db, channel_type, 6.0,
                0.0, "none", g_node._tx_config.sps, rng,
            )
            frame = AsyncDataFrame(segment_type=0, data=payload)
            n_mac = len(frame.pack())
            rx = t_node.receive(rx_iq, n_mac)
            success = rx.success
            g_node.process_feedback(success)
            if not success:
                g_node._qos.arq.on_ack_received()
        else:
            g_node._qos.arq.on_ack_received()

        # MCS 自适应
        suggested = g_node.recommended_mcs
        if suggested != g_node.config.mcs_index:
            g_node.update_mcs(suggested)
            t_node.update_mcs(suggested)

        frame_idx.append(i)
        mcs_hist.append(g_node.config.mcs_index)
        fer_hist.append(g_node.stats["fer"])
        success_hist.append(success)

    return {
        "frame_idx": frame_idx,
        "mcs_history": mcs_hist,
        "fer_history": fer_hist,
        "success_history": success_hist,
    }


def _run_dual_frames(
    tx_node, rx_node,
    snr_range_db: np.ndarray,
    n_frames: int,
    payload_size: int,
    rng: np.random.Generator,
    channel_type: str,
) -> list[float]:
    """内部: 对多个 SNR 运行帧传输, 返回 FER 列表。"""
    from nearlink_sdr.mac.frame import AsyncDataFrame

    fer_list = []
    for snr in snr_range_db:
        errors = 0
        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            tx_node.send(payload)
            tx = tx_node.transmit()
            if tx.iq is None:
                errors += 1
                tx_node._qos.arq.on_ack_received()
                continue
            rx_iq = _channel_impair(
                tx.iq, float(snr), channel_type, 6.0,
                0.0, "none", tx_node._tx_config.sps, rng,
            )
            frame = AsyncDataFrame(segment_type=0, data=payload)
            n_mac = len(frame.pack())
            rx = rx_node.receive(rx_iq, n_mac)
            tx_node.process_feedback(rx.success)
            if not rx.success:
                tx_node._qos.arq.on_ack_received()
                errors += 1
        fer_list.append(errors / n_frames)
    return fer_list


def run_phase14_simulation() -> None:
    """Phase 14: 双节点端到端仿真可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr = np.arange(0, 16, 2)

    print("Phase 14.1 - 双节点数据交换 ...")
    r1 = sim_dual_node_link(snr_range_db=snr, n_frames=30)
    print(f"  {'SNR':>5s}  {'FER':>10s}  {'ByteBER':>10s}")
    for s, f, b in zip(r1["snr_db"], r1["fer"], r1["byte_ber"]):
        print(f"  {s:5.0f}  {f:10.4f}  {b:10.6f}")

    print("Phase 14.2 - 双节点安全通信 ...")
    r2 = sim_dual_node_secure_link(snr_range_db=snr, n_frames=30)

    print("Phase 14.3 - MCS 自适应 ...")
    r3 = sim_dual_node_mcs_adapt(snr_db=8.0, n_frames=60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 14 - Dual Node E2E Simulation", fontsize=14, fontweight="bold")

    # 14.1 FER 曲线
    ax = axes[0, 0]
    ax.semilogy(r1["snr_db"], np.maximum(r1["fer"], 1e-4), "b-o", label="FER")
    ax.semilogy(r1["snr_db"], np.maximum(r1["byte_ber"], 1e-6), "r--s", label="Byte BER")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Error Rate")
    ax.set_title("14.1 - Dual Node FER/BER")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    # 14.2 明文 vs 加密
    ax = axes[0, 1]
    ax.semilogy(r2["snr_db"], np.maximum(r2["fer_plain"], 1e-4), "b-o", label="Plain")
    ax.semilogy(r2["snr_db"], np.maximum(r2["fer_encrypted"], 1e-4), "r--s", label="Encrypted")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    pairing_str = "OK" if r2["pairing_ok"] else "FAIL"
    ax.set_title(f"14.2 - Plain vs Encrypted (Pairing: {pairing_str})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    # 14.3 MCS 自适应
    ax = axes[1, 0]
    ax.plot(r3["frame_idx"], r3["mcs_history"], "b-", label="MCS")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("MCS Index")
    ax.set_title("14.3 - MCS Adaptation")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    # 14.4 FER 与成功率
    ax = axes[1, 1]
    ax.plot(r3["frame_idx"], r3["fer_history"], "r-", label="FER", alpha=0.7)
    sc = [i for i, s in zip(r3["frame_idx"], r3["success_history"], strict=False) if s]
    ax.scatter(sc, [0.05] * len(sc), c="green", s=5, label="TX OK")
    fc = [i for i, s in zip(r3["frame_idx"], r3["success_history"], strict=False) if not s]
    ax.scatter(fc, [0.05] * len(fc), c="red", s=5, label="TX FAIL")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("FER")
    ax.set_title("14.4 - FER & TX Success")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase14.png", dpi=150)
    print("\nPhase 14 curves saved to ber_phase14.png")


# ── Phase 15: SleNode 集成仿真 ──


def sim_node_hopping_link(
    snr_db: float = 10.0,
    n_frames: int = 100,
    mcs_index: int = 7,
    payload_size: int = 10,
    seed: int = 42,
) -> dict:
    """SleNode 跳频数据链路仿真。

    G 节点和 T 节点在跳频序列上逐帧通信,
    每帧发送后推进时隙, 观察信道号变化和传输成功率。

    :returns: {
            "channels": 使用过的信道号列表,
            "unique_channels": 唯一信道数,
            "success_count": 成功帧数,
            "fer": 帧错误率,
            "tx_powers": 各帧发射功率列表,
        }
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.link_manager import Role
    from nearlink_sdr.node import NodeConfig, NodeRole, SleNode

    rng = np.random.default_rng(seed)

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    g_node = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=mcs_index, max_retransmit=0,
    ))
    t_node = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=mcs_index, max_retransmit=0,
    ))

    g_node.start_advertising()
    t_node.start_scanning()
    t_node.connect(g_addr)
    g_node.accept_connection(t_addr, Role.G_NODE)

    channels = []
    successes = 0
    tx_powers = []

    for _ in range(n_frames):
        payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
        g_node.send(payload)
        tx = g_node.transmit()

        if tx.iq is None:
            g_node._qos.arq.on_ack_received()
            channels.append(-1)
            tx_powers.append(g_node.tx_power_dbm)
            g_node.advance_slot(1)
            t_node.advance_slot(1)
            continue

        channels.append(tx.channel)
        tx_powers.append(g_node.tx_power_dbm)

        # 信道传输 (理想信道, 仅关注跳频流程)
        ch = ChannelModel(snr_db=snr_db)
        rx_iq = ch.apply_awgn(tx.iq, g_node._tx_config.sps)

        frame = AsyncDataFrame(segment_type=0, data=payload)
        n_mac = len(frame.pack())
        rx = t_node.receive(rx_iq, n_mac)
        g_node.process_feedback(rx.success)

        if rx.success:
            successes += 1
        else:
            g_node._qos.arq.on_ack_received()

        g_node.advance_slot(1)
        t_node.advance_slot(1)

    unique_ch = len({c for c in channels if c >= 0})
    return {
        "channels": channels,
        "unique_channels": unique_ch,
        "success_count": successes,
        "fer": 1.0 - successes / max(n_frames, 1),
        "tx_powers": tx_powers,
    }


def sim_node_access_flow(seed: int = 42) -> dict:
    """SleNode 完整接入流程仿真。

    模拟 G 节点广播 → T 节点扫描/发现 → 接入 → 数据交换 → 断开。

    :returns: {
            "g_state": G 节点最终状态,
            "t_state": T 节点最终状态,
            "broadcast_frame_valid": 广播帧是否有效,
            "data_roundtrip_ok": 数据往返是否成功,
            "scheduler_active": 调度器是否已注册链路,
            "disconnect_ok": 断开是否成功,
        }
    """
    from nearlink_sdr.mac.link_manager import Role
    from nearlink_sdr.node import NodeConfig, NodeRole, NodeState, SleNode

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    g_node = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=5, max_retransmit=0,
    ))
    t_node = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=5, max_retransmit=0,
    ))

    # 1. 广播
    adv_frame = g_node.start_advertising()
    broadcast_valid = adv_frame is not None

    # 2. 扫描 + 接入
    t_node.start_scanning()
    t_node.connect(g_addr)
    g_node.accept_connection(t_addr, Role.G_NODE)

    scheduler_active = 0 in g_node.scheduler.event_schedulers

    # 3. 数据交换
    payload = b"integration_test_data"
    g_node.send(payload)
    tx = g_node.transmit()
    roundtrip_ok = False
    if tx.iq is not None:
        from nearlink_sdr.mac.frame import AsyncDataFrame
        frame = AsyncDataFrame(segment_type=0, data=payload)
        rx = t_node.receive(tx.iq, len(frame.pack()))
        roundtrip_ok = rx.success and rx.data == payload

    # 4. 断开
    g_node.disconnect()
    disconnect_ok = g_node.state == NodeState.DISCONNECTED

    return {
        "g_state": g_node.state.name,
        "t_state": t_node.state.name,
        "broadcast_frame_valid": broadcast_valid,
        "data_roundtrip_ok": roundtrip_ok,
        "scheduler_active": scheduler_active,
        "disconnect_ok": disconnect_ok,
    }


def sim_node_channel_sweep(
    snr_range_db: np.ndarray | None = None,
    n_frames: int = 50,
    mcs_index: int = 7,
    payload_size: int = 10,
    channel_type: str = "awgn",
    seed: int = 42,
) -> dict:
    """SleNode 内置信道模型扫频仿真。

    使用节点内置的 ChannelModel, 遍历 SNR 范围统计 FER。

    :returns: {"snr_db": [...], "fer": [...], "mcs_history": [...]}
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.link_manager import Role
    from nearlink_sdr.node import NodeConfig, NodeRole, SleNode
    from nearlink_sdr.phy.channel import ChannelConfig

    if snr_range_db is None:
        snr_range_db = np.arange(0, 16, 2)

    rng = np.random.default_rng(seed)

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    fer_list = []
    mcs_history = []

    for snr in snr_range_db:
        g_node = SleNode(config=NodeConfig(
            address=g_addr, role=NodeRole.G_NODE,
            frame_type=2, mcs_index=mcs_index, max_retransmit=0,
        ))
        t_node = SleNode(config=NodeConfig(
            address=t_addr, role=NodeRole.T_NODE,
            frame_type=2, mcs_index=mcs_index, max_retransmit=0,
            channel_config=ChannelConfig(
                snr_db=float(snr), channel_type=channel_type,
            ),
        ))

        g_node.start_advertising()
        t_node.start_scanning()
        t_node.connect(g_addr)
        g_node.accept_connection(t_addr, Role.G_NODE)

        errors = 0
        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            g_node.send(payload)
            tx = g_node.transmit()
            if tx.iq is None:
                errors += 1
                g_node._qos.arq.on_ack_received()
                continue

            frame = AsyncDataFrame(segment_type=0, data=payload)
            rx = t_node.receive(tx.iq, len(frame.pack()))
            g_node.process_feedback(rx.success)
            if not rx.success:
                errors += 1
                g_node._qos.arq.on_ack_received()

        fer_list.append(errors / n_frames)
        mcs_history.append(g_node.config.mcs_index)

    return {
        "snr_db": snr_range_db.tolist(),
        "fer": fer_list,
        "mcs_history": mcs_history,
    }


def sim_node_power_adapt(
    snr_db: float = 8.0,
    n_frames: int = 80,
    payload_size: int = 10,
    seed: int = 42,
) -> dict:
    """SleNode 功率自适应仿真。

    节点逐帧通信, 根据反馈结果动态调整发射功率。

    :returns: {
            "frame_idx": [...],
            "power_history": [...],
            "success_history": [...],
            "fer": float,
        }
    """
    from nearlink_sdr.mac.frame import AsyncDataFrame
    from nearlink_sdr.mac.link_manager import Role
    from nearlink_sdr.node import NodeConfig, NodeRole, SleNode

    rng = np.random.default_rng(seed)

    g_addr = b"\x01\x02\x03\x04\x05\x06"
    t_addr = b"\x0A\x0B\x0C\x0D\x0E\x0F"

    g_node = SleNode(config=NodeConfig(
        address=g_addr, role=NodeRole.G_NODE,
        frame_type=2, mcs_index=7, max_retransmit=0,
        tx_power_dbm=0.0, max_power_dbm=15.0, min_power_dbm=-10.0,
    ))
    t_node = SleNode(config=NodeConfig(
        address=t_addr, role=NodeRole.T_NODE,
        frame_type=2, mcs_index=7, max_retransmit=0,
    ))

    g_node.start_advertising()
    t_node.start_scanning()
    t_node.connect(g_addr)
    g_node.accept_connection(t_addr, Role.G_NODE)

    power_history = []
    success_history = []
    consecutive_fails = 0

    for _i in range(n_frames):
        payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
        g_node.send(payload)
        tx = g_node.transmit()

        if tx.iq is None:
            g_node._qos.arq.on_ack_received()
            power_history.append(g_node.tx_power_dbm)
            success_history.append(False)
            continue

        ch = ChannelModel(snr_db=snr_db)
        rx_iq = ch.apply_awgn(tx.iq, g_node._tx_config.sps)

        frame = AsyncDataFrame(segment_type=0, data=payload)
        rx = t_node.receive(rx_iq, len(frame.pack()))
        g_node.process_feedback(rx.success)

        if not rx.success:
            g_node._qos.arq.on_ack_received()
            consecutive_fails += 1
            if consecutive_fails >= 3:
                g_node.adjust_power(2.0)
                consecutive_fails = 0
        else:
            consecutive_fails = 0

        power_history.append(g_node.tx_power_dbm)
        success_history.append(rx.success)

    fer = 1.0 - sum(success_history) / max(len(success_history), 1)
    return {
        "frame_idx": list(range(n_frames)),
        "power_history": power_history,
        "success_history": success_history,
        "fer": fer,
    }


def sim_node_measurement(
    n_measur: int = 64,
    seed: int = 42,
) -> dict:
    """SleNode 测量信号生成仿真。

    :returns: {
            "signal_length": 测量信号长度,
            "signal_energy": 信号能量,
        }
    """
    from nearlink_sdr.node import NodeConfig, SleNode

    node = SleNode(config=NodeConfig(address=b"\x01" * 6))
    sig = node.generate_measurement_signal(n_measur=n_measur)

    return {
        "signal_length": len(sig),
        "signal_energy": float(np.sum(np.abs(sig) ** 2)),
    }


def run_phase15_simulation() -> None:
    """Phase 15: SleNode 集成仿真可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Phase 15.1 - 节点接入流程 ...")
    r1 = sim_node_access_flow()
    print(f"  广播帧: {'OK' if r1['broadcast_frame_valid'] else 'FAIL'}")
    print(f"  数据往返: {'OK' if r1['data_roundtrip_ok'] else 'FAIL'}")
    print(f"  调度注册: {'OK' if r1['scheduler_active'] else 'FAIL'}")
    print(f"  断开连接: {'OK' if r1['disconnect_ok'] else 'FAIL'}")

    print("\nPhase 15.2 - 跳频链路 ...")
    r2 = sim_node_hopping_link(snr_db=12.0, n_frames=100)
    print(f"  唯一信道数: {r2['unique_channels']}")
    print(f"  FER: {r2['fer']:.3f}")

    print("\nPhase 15.3 - 信道扫频 ...")
    snr = np.arange(0, 16, 2)
    r3 = sim_node_channel_sweep(snr_range_db=snr, n_frames=30)

    print("\nPhase 15.4 - 功率自适应 ...")
    r4 = sim_node_power_adapt(snr_db=-2.0, n_frames=60)
    print(f"  FER: {r4['fer']:.3f}")
    print(f"  功率范围: [{min(r4['power_history']):.1f}, "
          f"{max(r4['power_history']):.1f}] dBm")

    print("\nPhase 15.5 - 测量信号 ...")
    r5 = sim_node_measurement()
    print(f"  信号长度: {r5['signal_length']}")
    print(f"  信号能量: {r5['signal_energy']:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 15 - SleNode Integration Simulation", fontsize=14, fontweight="bold")

    # 15.2 跳频信道分布
    ax = axes[0, 0]
    valid_ch = [c for c in r2["channels"] if c >= 0]
    if valid_ch:
        ax.hist(valid_ch, bins=range(min(valid_ch), max(valid_ch) + 2), edgecolor="black")
    ax.set_xlabel("Channel Number")
    ax.set_ylabel("Count")
    ax.set_title(f"15.2 - Hopping Channel Distribution ({r2['unique_channels']} channels)")
    ax.grid(True, ls="--", alpha=0.3)

    # 15.3 信道扫频 FER
    ax = axes[0, 1]
    ax.semilogy(r3["snr_db"], np.maximum(r3["fer"], 1e-4), "b-o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("15.3 - Node Channel Sweep FER")
    ax.grid(True, ls="--", alpha=0.3)

    # 15.4 功率自适应
    ax = axes[1, 0]
    ax.plot(r4["frame_idx"], r4["power_history"], "b-", label="TX Power")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("TX Power (dBm)")
    ax.set_title(f"15.4 - Power Adaptation (FER={r4['fer']:.3f})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    # 15.4 成功/失败标记
    ax2 = axes[1, 1]
    sc = [i for i, s in zip(r4["frame_idx"], r4["success_history"], strict=False) if s]
    fc = [i for i, s in zip(r4["frame_idx"], r4["success_history"], strict=False) if not s]
    ax2.scatter(sc, [1] * len(sc), c="green", s=10, label="TX OK")
    ax2.scatter(fc, [0] * len(fc), c="red", s=10, label="TX FAIL")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Success")
    ax2.set_title("15.5 - Frame Success/Failure")
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["FAIL", "OK"])

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase15.png", dpi=150)
    print("\nPhase 15 curves saved to ber_phase15.png")


# ── Phase 16: 多用户干扰 + Doppler 时变信道 ──


def sim_doppler_link(
    doppler_range_hz: np.ndarray | None = None,
    snr_db: float = 12.0,
    n_frames: int = 100,
    mcs_index: int = 7,
    payload_size: int = 20,
    channel_type: str = "rayleigh",
    eq_method: str = "zf",
    seed: int = 42,
) -> dict:
    """Doppler 时变信道仿真: 扫描不同多普勒频移下的 FER。

    使用 Jakes 求和正弦模型生成时间相关衰落, 评估 FER 随 Doppler 扩展的变化。

    :param doppler_range_hz: Doppler 频移扫描范围 (Hz)。
    :param snr_db: 固定 SNR (dB)。
    :param n_frames: 每个 Doppler 点仿真帧数。
    :param mcs_index: MCS 索引。
    :param payload_size: 载荷字节数。
    :param channel_type: 衰落类型 ("rayleigh" / "rician")。
    :param eq_method: 均衡方法。
    :param seed: 随机种子。

    :returns: {"doppler_hz": [...], "fer": [...], "avg_fade_depth_db": [...]}
    """
    from nearlink_sdr.phy.channel import ChannelConfig, ChannelModel
    from nearlink_sdr.phy.equalizer import equalize_1tap
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if doppler_range_hz is None:
        doppler_range_hz = np.array([0, 5, 10, 20, 50, 100, 200])

    rng = np.random.default_rng(seed)
    cfg = TxConfig(
        frame_type=2, mcs_index=mcs_index, pid=0x123456,
        whitening_seed=0x52, crc_seed=0x555555, crc_len=24,
        ctrl_bits_len=28, pilot_interval=8,
    )

    fer_list = []
    fade_depth_list = []

    for fd in doppler_range_hz:
        errors = 0
        fade_depths = []

        for _i in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            from nearlink_sdr.mac.frame import AsyncDataFrame
            frame = AsyncDataFrame(segment_type=0, data=payload)
            mac_bytes = frame.pack()

            iq = mac_to_iq(mac_bytes, cfg)

            ch_cfg = ChannelConfig(
                snr_db=snr_db,
                channel_type=channel_type,
                max_doppler_hz=float(fd),
                symbol_rate_hz=cfg.sps * cfg.symbol_rate_mhz * 1e6,
                seed=int(rng.integers(0, 2**31)),
            )
            ch = ChannelModel(config=ch_cfg)
            rx_iq = ch.apply_fading(iq)

            # 衰落深度统计
            taps = ch.last_taps
            if taps is not None:
                h = taps[0, :]
                fade_db = 20 * np.log10(np.abs(h) + 1e-20)
                fade_depths.append(float(np.mean(fade_db)))

            # 均衡
            if eq_method != "none" and taps is not None:
                h = taps[0, :]
                rx_iq = equalize_1tap(rx_iq, h, ch.noise_variance, method=eq_method)

            rx = iq_to_mac(rx_iq, cfg, len(mac_bytes))
            if not rx.crc_ok:
                errors += 1
            else:
                try:
                    recovered = AsyncDataFrame.unpack(rx.mac_payload)
                    if recovered.data != payload:
                        errors += 1
                except (ValueError, IndexError):
                    errors += 1

        fer_list.append(errors / n_frames)
        fade_depth_list.append(np.mean(fade_depths) if fade_depths else 0.0)

    return {
        "doppler_hz": [float(d) for d in doppler_range_hz],
        "fer": fer_list,
        "avg_fade_depth_db": fade_depth_list,
    }


def sim_multi_user_interference(
    n_interferers_range: list[int] | None = None,
    sir_db: float = 10.0,
    snr_db: float = 15.0,
    n_frames: int = 100,
    mcs_index: int = 7,
    payload_size: int = 20,
    channel_type: str = "awgn",
    freq_offset_hz: float = 0.0,
    seed: int = 42,
) -> dict:
    """多用户干扰仿真: 扫描不同干扰用户数下的 FER 和 SINR。

    多个干扰用户在同一频段 (或邻信道) 独立发送, 叠加到有用信号上。

    :param n_interferers_range: 干扰用户数扫描列表。
    :param sir_db: 每个干扰用户的 SIR (dB)。
    :param snr_db: 噪声 SNR (dB)。
    :param n_frames: 每个点仿真帧数。
    :param mcs_index: MCS 索引。
    :param payload_size: 载荷字节数。
    :param channel_type: 信道类型。
    :param freq_offset_hz: 干扰频偏 (Hz), 0 为同信道。
    :param seed: 随机种子。

    :returns: {"n_interferers": [...], "fer": [...], "sinr_db": [...]}
    """
    from nearlink_sdr.phy.channel import (
        ChannelConfig,
        ChannelModel,
        InterferenceConfig,
        add_interference,
        compute_sinr,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if n_interferers_range is None:
        n_interferers_range = [0, 1, 2, 3, 5, 8]

    rng = np.random.default_rng(seed)
    cfg = TxConfig(
        frame_type=2, mcs_index=mcs_index, pid=0x123456,
        whitening_seed=0x52, crc_seed=0x555555, crc_len=24,
        ctrl_bits_len=28, pilot_interval=8,
    )
    sample_rate = cfg.sps * cfg.symbol_rate_mhz * 1e6

    fer_list = []
    sinr_list = []

    for n_intf in n_interferers_range:
        errors = 0
        sinr_acc = []

        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            from nearlink_sdr.mac.frame import AsyncDataFrame
            frame = AsyncDataFrame(segment_type=0, data=payload)
            mac_bytes = frame.pack()

            iq = mac_to_iq(mac_bytes, cfg)

            # 应用信道
            ch_cfg = ChannelConfig(
                snr_db=snr_db,
                channel_type=channel_type,
                seed=int(rng.integers(0, 2**31)),
            )
            ch = ChannelModel(config=ch_cfg)
            noisy_iq = ch.apply_fading(iq)

            # 叠加干扰
            if n_intf > 0:
                intfs = [
                    InterferenceConfig(
                        sir_db=sir_db,
                        freq_offset_hz=freq_offset_hz,
                        seed=int(rng.integers(0, 2**31)),
                    )
                    for _ in range(n_intf)
                ]
                rx_iq = add_interference(noisy_iq, intfs, sample_rate, rng)
                sinr_acc.append(compute_sinr(iq, noisy_iq, rx_iq))
            else:
                rx_iq = noisy_iq
                # 无干扰时 SINR ≈ SNR
                sinr_acc.append(snr_db)

            rx = iq_to_mac(rx_iq, cfg, len(mac_bytes))
            if not rx.crc_ok:
                errors += 1
            else:
                try:
                    recovered = AsyncDataFrame.unpack(rx.mac_payload)
                    if recovered.data != payload:
                        errors += 1
                except (ValueError, IndexError):
                    errors += 1

        fer_list.append(errors / n_frames)
        sinr_list.append(float(np.mean(sinr_acc)))

    return {
        "n_interferers": list(n_interferers_range),
        "fer": fer_list,
        "sinr_db": sinr_list,
    }


def sim_sir_sweep(
    sir_range_db: np.ndarray | None = None,
    n_interferers: int = 1,
    snr_db: float = 20.0,
    n_frames: int = 100,
    mcs_index: int = 7,
    payload_size: int = 20,
    channel_type: str = "awgn",
    seed: int = 42,
) -> dict:
    """SIR 扫描仿真: 固定干扰用户数, 扫描 SIR 下的 FER。

    :param sir_range_db: SIR 扫描范围 (dB)。
    :param n_interferers: 干扰用户数。
    :param snr_db: 噪声 SNR (dB)。
    :param n_frames: 每个 SIR 点帧数。
    :param mcs_index: MCS 索引。
    :param payload_size: 载荷字节数。
    :param channel_type: 信道类型。
    :param seed: 随机种子。

    :returns: {"sir_db": [...], "fer": [...], "sinr_db": [...]}
    """
    from nearlink_sdr.phy.channel import (
        ChannelConfig,
        ChannelModel,
        InterferenceConfig,
        add_interference,
    )
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if sir_range_db is None:
        sir_range_db = np.arange(-10, 25, 5)

    rng = np.random.default_rng(seed)
    cfg = TxConfig(
        frame_type=2, mcs_index=mcs_index, pid=0x123456,
        whitening_seed=0x52, crc_seed=0x555555, crc_len=24,
        ctrl_bits_len=28, pilot_interval=8,
    )
    sample_rate = cfg.sps * cfg.symbol_rate_mhz * 1e6

    fer_list = []
    sinr_list = []

    for sir in sir_range_db:
        errors = 0
        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            from nearlink_sdr.mac.frame import AsyncDataFrame
            frame = AsyncDataFrame(segment_type=0, data=payload)
            mac_bytes = frame.pack()

            iq = mac_to_iq(mac_bytes, cfg)

            ch_cfg = ChannelConfig(
                snr_db=snr_db,
                channel_type=channel_type,
                seed=int(rng.integers(0, 2**31)),
            )
            ch = ChannelModel(config=ch_cfg)
            noisy_iq = ch.apply_fading(iq)

            intfs = [
                InterferenceConfig(
                    sir_db=float(sir),
                    seed=int(rng.integers(0, 2**31)),
                )
                for _ in range(n_interferers)
            ]
            rx_iq = add_interference(noisy_iq, intfs, sample_rate, rng)

            rx = iq_to_mac(rx_iq, cfg, len(mac_bytes))
            if not rx.crc_ok:
                errors += 1
            else:
                try:
                    recovered = AsyncDataFrame.unpack(rx.mac_payload)
                    if recovered.data != payload:
                        errors += 1
                except (ValueError, IndexError):
                    errors += 1

        fer = errors / n_frames
        fer_list.append(fer)
        # 理论 SINR ≈ (S / (N + I)) -> S/(S/SNR + n_intf*S/SIR)
        snr_lin = 10 ** (snr_db / 10)
        sir_lin = 10 ** (float(sir) / 10)
        sinr_lin = 1.0 / (1.0 / snr_lin + n_interferers / sir_lin)
        sinr_list.append(float(10 * np.log10(sinr_lin)))

    return {
        "sir_db": [float(s) for s in sir_range_db],
        "fer": fer_list,
        "sinr_db": sinr_list,
    }


def sim_doppler_multipath_link(
    doppler_range_hz: np.ndarray | None = None,
    snr_db: float = 15.0,
    n_frames: int = 80,
    mcs_index: int = 7,
    payload_size: int = 20,
    seed: int = 42,
) -> dict:
    """Doppler + 多径信道联合仿真。

    在 ITU Indoor Office B 多径模型基础上叠加 Doppler 时变,
    评估信道时变对频率选择性衰落下的接收性能影响。

    :returns: {"doppler_hz": [...], "fer": [...]}
    """
    from nearlink_sdr.phy.channel import PDP_INDOOR_OFFICE, ChannelConfig, ChannelModel
    from nearlink_sdr.phy.equalizer import equalize_mmse_freq
    from nearlink_sdr.phy.mac_interface import iq_to_mac, mac_to_iq
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    if doppler_range_hz is None:
        doppler_range_hz = np.array([0, 5, 10, 20, 50, 100])

    rng = np.random.default_rng(seed)
    cfg = TxConfig(
        frame_type=2, mcs_index=mcs_index, pid=0x123456,
        whitening_seed=0x52, crc_seed=0x555555, crc_len=24,
        ctrl_bits_len=28, pilot_interval=8,
    )

    fer_list = []
    for fd in doppler_range_hz:
        errors = 0
        for _ in range(n_frames):
            payload = bytes(rng.integers(0, 256, payload_size, dtype=np.uint8))
            from nearlink_sdr.mac.frame import AsyncDataFrame
            frame = AsyncDataFrame(segment_type=0, data=payload)
            mac_bytes = frame.pack()

            iq = mac_to_iq(mac_bytes, cfg)

            ch_cfg = ChannelConfig(
                snr_db=snr_db,
                channel_type="multipath",
                pdp=list(PDP_INDOOR_OFFICE),
                max_doppler_hz=float(fd),
                symbol_rate_hz=cfg.sps * cfg.symbol_rate_mhz * 1e6,
                seed=int(rng.integers(0, 2**31)),
            )
            ch = ChannelModel(config=ch_cfg)
            rx_iq = ch.apply_fading(iq)

            # MMSE 频域均衡 (用首符号抽头近似)
            taps = ch.last_taps
            if taps is not None:
                h_time = taps[:, 0]
                h_freq = np.fft.fft(h_time, len(rx_iq))
                rx_iq = equalize_mmse_freq(rx_iq, h_freq, ch.noise_variance)

            rx = iq_to_mac(rx_iq, cfg, len(mac_bytes))
            if not rx.crc_ok:
                errors += 1
            else:
                try:
                    recovered = AsyncDataFrame.unpack(rx.mac_payload)
                    if recovered.data != payload:
                        errors += 1
                except (ValueError, IndexError):
                    errors += 1

        fer_list.append(errors / n_frames)

    return {
        "doppler_hz": [float(d) for d in doppler_range_hz],
        "fer": fer_list,
    }


def run_phase16_simulation(n_frames: int = 50) -> None:
    """Phase 16: 多用户干扰 + Doppler 时变信道仿真可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Phase 16.1 - Doppler 时变信道 FER ...")
    r1 = sim_doppler_link(
        doppler_range_hz=np.array([0, 5, 10, 20, 50, 100]),
        snr_db=12.0, n_frames=n_frames,
    )
    for fd, fer in zip(r1["doppler_hz"], r1["fer"], strict=False):
        print(f"  f_d={fd:5.0f} Hz  FER={fer:.3f}")

    print("\nPhase 16.2 - 多用户干扰 (SIR 扫描) ...")
    r2 = sim_sir_sweep(
        sir_range_db=np.arange(-5, 25, 5),
        n_interferers=1, snr_db=20.0, n_frames=n_frames,
    )
    for sir, fer in zip(r2["sir_db"], r2["fer"], strict=False):
        print(f"  SIR={sir:5.0f} dB  FER={fer:.3f}")

    print("\nPhase 16.3 - 多干扰用户数扫描 ...")
    r3 = sim_multi_user_interference(
        n_interferers_range=[0, 1, 2, 3, 5],
        sir_db=10.0, snr_db=15.0, n_frames=n_frames,
    )
    for ni, fer, sinr in zip(
        r3["n_interferers"], r3["fer"], r3["sinr_db"], strict=False,
    ):
        print(f"  N_intf={ni}  FER={fer:.3f}  SINR={sinr:.1f} dB")

    print("\nPhase 16.4 - Doppler + 多径 ...")
    r4 = sim_doppler_multipath_link(
        doppler_range_hz=np.array([0, 5, 10, 20, 50]),
        snr_db=15.0, n_frames=n_frames,
    )
    for fd, fer in zip(r4["doppler_hz"], r4["fer"], strict=False):
        print(f"  f_d={fd:5.0f} Hz  FER={fer:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 16 - Interference & Doppler Simulation", fontsize=14, fontweight="bold")

    # 16.1 Doppler FER
    ax = axes[0, 0]
    ax.semilogy(r1["doppler_hz"], np.maximum(r1["fer"], 1e-4), "b-o")
    ax.set_xlabel("Doppler Spread (Hz)")
    ax.set_ylabel("FER")
    ax.set_title("16.1 - Doppler Time-Varying Channel FER")
    ax.grid(True, ls="--", alpha=0.3)

    # 16.2 SIR sweep
    ax = axes[0, 1]
    ax.semilogy(r2["sir_db"], np.maximum(r2["fer"], 1e-4), "r-s")
    ax.set_xlabel("SIR (dB)")
    ax.set_ylabel("FER")
    ax.set_title("16.2 - SIR Sweep (1 Interferer)")
    ax.grid(True, ls="--", alpha=0.3)

    # 16.3 Multi-user
    ax = axes[1, 0]
    ax.plot(r3["n_interferers"], r3["fer"], "g-^", label="FER")
    ax_twin = ax.twinx()
    ax_twin.plot(r3["n_interferers"], r3["sinr_db"], "m--o", label="SINR")
    ax.set_xlabel("Number of Interferers")
    ax.set_ylabel("FER")
    ax_twin.set_ylabel("SINR (dB)")
    ax.set_title("16.3 - Multi-User Interference")
    ax.grid(True, ls="--", alpha=0.3)

    # 16.4 Doppler + Multipath
    ax = axes[1, 1]
    ax.semilogy(r4["doppler_hz"], np.maximum(r4["fer"], 1e-4), "k-D")
    ax.set_xlabel("Doppler Spread (Hz)")
    ax.set_ylabel("FER")
    ax.set_title("16.4 - Doppler + Multipath (Indoor Office)")
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(_ensure_output_dir() / "ber_phase16.png", dpi=150)
    print("\nPhase 16 curves saved to ber_phase16.png")


if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    _dispatch = {
        "phase1": run_phase1_simulation,
        "phase2": run_phase2_simulation,
        "phase3": run_phase3_simulation,
        "phase4": run_phase4_simulation,
        "phase5": run_phase5_simulation,
        "phase6": run_phase6_simulation,
        "phase7": run_phase7_simulation,
        "phase8": run_phase8_simulation,
        "phase9": run_phase9_simulation,
        "phase10": run_phase10_simulation,
        "phase11": run_phase11_simulation,
        "phase12": run_phase12_simulation,
        "phase13": run_phase13_simulation,
        "phase14": run_phase14_simulation,
        "phase15": run_phase15_simulation,
        "phase16": run_phase16_simulation,
    }
    _dispatch.get(phase, run_phase1_simulation)()
