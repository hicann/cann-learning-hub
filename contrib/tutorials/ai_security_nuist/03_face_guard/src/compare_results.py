# -*- coding: utf-8 -*-
"""
第三阶段：汇总对比结果。

输出到 answer/comparison/：
  - compare_grid.png   baseline / step_0250 / step_0500 / step_1000 / step_2000 横向拼接
  - loss_curve.png     微调 loss 曲线 (loss_log.csv)，若 matplotlib 可用
  - summary.json       各 step 是否存在
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_DIR, OUTPUT_DIR, log  # noqa: E402

STEPS = ["baseline", "step_0250", "step_0500", "step_1000", "step_2000"]
COMPARE_DIR = OUTPUT_DIR / "comparison"


def grid_path_for(step: str) -> Path:
    if step == "baseline":
        return OUTPUT_DIR / "baseline" / "baseline_grid.png"
    n = int(step.split("_")[1])
    return OUTPUT_DIR / f"step_{n:04d}" / "grid.png"


def build_compare_grid() -> dict:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    summary = {}
    for step in STEPS:
        p = grid_path_for(step)
        summary[step] = p.exists()
        if p.exists():
            panels.append((step, Image.open(p).convert("RGB")))
        else:
            panels.append((step, None))

    valid = [(s, img) for s, img in panels if img is not None]
    if not valid:
        log("no grid images found, skip compare_grid.png")
        return summary

    # 统一高度后横向拼接
    h = min(img.height for _, img in valid)
    resized = []
    for s, img in valid:
        if img.height != h:
            w = int(img.width * h / img.height)
            img = img.resize((w, h), Image.LANCZOS)
        resized.append(img)
    total_w = sum(img.width for img in resized) + 10 * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, h + 30), (255, 255, 255))
    x = 0
    for (s, _), img in zip(valid, resized):
        canvas.paste(img, (x, 30))
        x += img.width + 10
    out = COMPARE_DIR / "compare_grid.png"
    canvas.save(out)
    log(f"saved compare_grid -> {out}  ({canvas.size})")
    return summary


def build_loss_curve() -> bool:
    loss_csv = OUTPUT_DIR / "loss_log.csv"
    if not loss_csv.exists():
        log("loss_log.csv not found, skip loss curve")
        return False
    rows = []
    with loss_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        log("loss_log.csv empty, skip loss curve")
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"matplotlib unavailable ({exc!r}), skip loss curve plot")
        return False

    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color in [("lossD_real", "tab:blue"), ("lossD_fake", "tab:orange"), ("lossG", "tab:green")]:
        if key in rows[0] and rows[0][key] not in (None, ""):
            vals = [float(r[key]) for r in rows]
            ax.plot(steps, vals, label=key, color=color, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("ProGAN finetune loss (AIGIBench)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = COMPARE_DIR / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log(f"saved loss_curve -> {out}  ({len(rows)} points)")
    return True


def main() -> int:
    summary = build_compare_grid()
    summary["loss_curve"] = build_loss_curve()
    out = COMPARE_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved summary -> {out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
