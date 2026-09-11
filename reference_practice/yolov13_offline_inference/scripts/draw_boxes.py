# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import logging
import os
import cv2

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s : %(message)s"
)

INPUT_SIZE = 640
PAD_COLOR = (114, 114, 114)
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]


def letterbox_coords(orig_w, orig_h):
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_l = (INPUT_SIZE - new_w) // 2
    pad_t = (INPUT_SIZE - new_h) // 2
    return scale, pad_l, pad_t


def draw(img_path, result_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.error("cannot read %s", img_path)
        return

    h, w = img.shape[:2]
    scale, pad_l, pad_t = letterbox_coords(w, h)

    with open(result_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        label, conf = parts[0], float(parts[1])
        x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])

        x1 = int((x1 - pad_l) / scale)
        y1 = int((y1 - pad_t) / scale)
        x2 = int((x2 - pad_l) / scale)
        y2 = int((y2 - pad_t) / scale)

        color = COLORS[hash(label) % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    cv2.imwrite(output_path, img)
    logging.info("saved: %s", output_path)


if __name__ == "__main__":
    import glob

    for fpath in glob.glob("*.jpg") + glob.glob("*.png"):
        basename = os.path.splitext(fpath)[0]
        result = basename + "_result.txt"
        if not os.path.exists(result):
            continue
        draw(fpath, result, basename + "_out.jpg")
