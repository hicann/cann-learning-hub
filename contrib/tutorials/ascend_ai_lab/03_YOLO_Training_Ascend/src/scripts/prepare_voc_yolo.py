#!/usr/bin/env python3
"""Convert PASCAL VOC XML annotations to YOLO txt labels for this Ascend lab.

Expected input after extracting VOC archives:
    /mnt/workspace/datasets/voc/
    └── VOCdevkit/
        ├── VOC2007/
        │   ├── JPEGImages/
        │   ├── Annotations/
        │   └── ImageSets/Main/
        └── VOC2012/
            ├── JPEGImages/
            ├── Annotations/
            └── ImageSets/Main/

Outputs:
    images/train2007/*.jpg
    images/train2012/*.jpg
    images/val2007/*.jpg
    labels/train2007/*.txt
    labels/train2012/*.txt
    labels/val2007/*.txt
    splits/train.txt
    splits/val.txt
    voc.names
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/workspace/datasets/voc", help="PASCAL VOC workspace root")
    parser.add_argument("--train-limit", type=int, default=0, help="0 means keep all VOC2007+VOC2012 trainval images")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means keep all VOC2007 test images")
    parser.add_argument("--seed", type=int, default=2026, help="random seed used when a limit is set")
    parser.add_argument("--copy-images", action="store_true", help="copy images instead of creating symlinks")
    parser.add_argument("--include-difficult", action="store_true", help="keep objects marked difficult in VOC XML")
    return parser.parse_args()


def voc_dir(root: Path, year: str) -> Path:
    candidates = [
        root / "VOCdevkit" / f"VOC{year}",
        root / f"VOC{year}",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"VOC{year} directory not found under {root}")


def read_ids(voc: Path, split: str) -> list[str]:
    split_path = voc / "ImageSets" / "Main" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    ids = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        item = line.strip().split()
        if item:
            ids.append(item[0])
    return ids


def limit_items(items: list[str], limit: int, seed: int) -> list[str]:
    if limit and 0 < limit < len(items):
        rng = random.Random(seed)
        return sorted(rng.sample(items, limit))
    return sorted(items)


def limit_pairs(items: list[tuple[str, str]], limit: int, seed: int) -> list[tuple[str, str]]:
    if limit and 0 < limit < len(items):
        rng = random.Random(seed)
        return sorted(rng.sample(items, limit))
    return sorted(items)


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def voc_bbox_to_yolo(box: ET.Element, width: int, height: int) -> tuple[float, float, float, float]:
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))

    xmin = max(0.0, min(xmin, width - 1.0))
    ymin = max(0.0, min(ymin, height - 1.0))
    xmax = max(0.0, min(xmax, width - 1.0))
    ymax = max(0.0, min(ymax, height - 1.0))

    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / width, cy / height, bw / width, bh / height


def convert_one_xml(xml_path: Path, label_path: Path, include_difficult: bool) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing size node in {xml_path}")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {xml_path}")

    rows = []
    class_to_id = {name: idx for idx, name in enumerate(VOC_CLASSES)}
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        difficult = int(obj.findtext("difficult", "0") or 0)
        if difficult and not include_difficult:
            continue
        if name not in class_to_id:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        cx, cy, bw, bh = voc_bbox_to_yolo(box, width, height)
        if bw <= 0 or bh <= 0:
            continue
        rows.append(f"{class_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def convert_split(
    root: Path,
    year: str,
    split: str,
    out_name: str,
    ids: list[str],
    copy_images: bool,
    include_difficult: bool,
) -> list[str]:
    voc = voc_dir(root, year)
    image_out = root / "images" / out_name
    label_out = root / "labels" / out_name
    rel_paths = []

    for image_id in ids:
        src_image = voc / "JPEGImages" / f"{image_id}.jpg"
        src_xml = voc / "Annotations" / f"{image_id}.xml"
        if not src_image.exists():
            print(f"skip missing image: {src_image}")
            continue
        if not src_xml.exists():
            print(f"skip missing annotation: {src_xml}")
            continue

        dst_image = image_out / f"{image_id}.jpg"
        dst_label = label_out / f"{image_id}.txt"
        link_or_copy(src_image, dst_image, copy_images)
        convert_one_xml(src_xml, dst_label, include_difficult)
        rel_paths.append(f"images/{out_name}/{image_id}.jpg")

    print(f"VOC{year} {split} -> {out_name}: {len(rel_paths)} images")
    return sorted(rel_paths)


def write_split(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(items)) + ("\n" if items else ""), encoding="utf-8")
    print(f"{path}: {len(items)} images")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    train_pairs = [("2007", image_id) for image_id in read_ids(voc_dir(root, "2007"), "trainval")]
    train_pairs += [("2012", image_id) for image_id in read_ids(voc_dir(root, "2012"), "trainval")]
    train_pairs = limit_pairs(train_pairs, args.train_limit, args.seed)
    train_2007 = [image_id for year, image_id in train_pairs if year == "2007"]
    train_2012 = [image_id for year, image_id in train_pairs if year == "2012"]
    try:
        val_2007 = limit_items(read_ids(voc_dir(root, "2007"), "test"), args.val_limit, args.seed)
        val_split = "test"
    except FileNotFoundError:
        val_2007 = limit_items(read_ids(voc_dir(root, "2007"), "val"), args.val_limit, args.seed)
        val_split = "val"

    train_items = []
    train_items += convert_split(root, "2007", "trainval", "train2007", train_2007, args.copy_images, args.include_difficult)
    train_items += convert_split(root, "2012", "trainval", "train2012", train_2012, args.copy_images, args.include_difficult)
    val_items = convert_split(root, "2007", val_split, "val2007", val_2007, args.copy_images, args.include_difficult)

    write_split(root / "splits" / "train.txt", train_items)
    write_split(root / "splits" / "val.txt", val_items)
    write_split(root / "splits" / "train2007.txt", [item for item in train_items if "/train2007/" in item])
    write_split(root / "splits" / "train2012.txt", [item for item in train_items if "/train2012/" in item])
    write_split(root / "splits" / "val2007.txt", val_items)
    (root / "voc.names").write_text("\n".join(VOC_CLASSES) + "\n", encoding="utf-8")

    print("PASCAL VOC YOLO labels are ready.")


if __name__ == "__main__":
    main()
