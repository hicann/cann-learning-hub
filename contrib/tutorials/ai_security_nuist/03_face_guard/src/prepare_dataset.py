# -*- coding: utf-8 -*-
"""
准备 data/aigibench_128/：从已构建的 AIGIBench-WFIR-ProGAN-1K 数据集
读取 256×256 人脸，缩放到 128×128，保存为 PNG。

输入默认：../../dataset/AIGIBench-WFIR-ProGAN-1K/images/  (可由 --src 覆盖)
输出：    data/aigibench_128/face_000001.png, face_000002.png, ...

纯 Pillow，无需 NPU/GPU，本地即可运行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
OUT_DIR = SRC_DIR / "data" / "aigibench_128"
OUT_SIZE = 128

SRC_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None,
                    help="源图像目录（256×256 人脸）；若未提供则报错。")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    args = ap.parse_args()

    if args.src is None:
        print("[prepare] ERROR: 必须用 --src 指定源图像目录。")
        print("        data/aigibench_128 已随项目提供，通常无需重新运行本脚本。")
        return 2
    if not args.src.exists():
        print(f"[prepare] ERROR: src not found: {args.src}")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in SRC_EXTS)
    if not files:
        print(f"[prepare] ERROR: no images in {args.src}")
        return 2

    print(f"[prepare] src={args.src}  count={len(files)}  out={args.out}  size={args.size}")

    n_ok = 0
    for i, p in enumerate(files, start=1):
        try:
            with Image.open(p) as img:
                img = img.convert("RGB").resize((args.size, args.size), Image.LANCZOS)
                img.save(args.out / f"face_{i:06d}.png", format="PNG")
            n_ok += 1
        except Exception as e:
            print(f"[prepare] WARN skip {p.name}: {e}")

    print(f"[prepare] done. wrote {n_ok} images to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
