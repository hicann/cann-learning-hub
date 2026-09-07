"""Shared Matplotlib settings for experiment 03 notebooks."""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


CLASS_COLORS = {
    "BENIGN": "#4C78A8",
    "FTP-Patator": "#F58518",
    "SSH-Patator": "#E45756",
}

CHANNEL_KEYS = [
    "packet_length",
    "flow_timing",
    "interaction",
    "tcp_flags",
    "active_idle",
    "supplemental",
]
CHANNEL_ORDER = CHANNEL_KEYS
CHANNEL_LABELS = ["包长/字节", "流时序", "交互统计", "TCP 标志", "活跃/空闲", "补充特征"]
CHANNEL_COLORS = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#B279A2", "#F2CF5B"]


def configure_plotting(chapter_dir: Path) -> Path:
    """Register the bundled Chinese font and apply the shared plot style."""
    font_path = chapter_dir / "src" / "fonts" / "NotoSansCJKsc-ExperimentSubset.otf"
    font_manager.fontManager.addfont(str(font_path))
    font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return font_path
