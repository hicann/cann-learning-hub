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

import os
import logging
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s : %(message)s"
)

INPUT_SIZE = 640
PAD_COLOR = (114, 114, 114)


def letterbox(image, target_size, pad_color):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    return (
        cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_h - pad_top,
            pad_left,
            pad_w - pad_left,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        ),
        scale,
        pad_left,
        pad_top,
    )


def process(input_path):
    try:
        img = cv2.imread(input_path)
        if img is None:
            logging.error("failed to read image: %s", input_path)
            return 1

        img_padded, scale, pad_left, pad_top = letterbox(img, INPUT_SIZE, PAD_COLOR)

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_f32 = img_rgb.astype("float32") / 255.0

        img_chw = img_f32.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_chw, 0).astype("float32")

        output_name = input_path.rsplit(".", 1)[0] + ".bin"
        img_batch.tofile(output_name)

        with open(input_path.rsplit(".", 1)[0] + "_info.txt", "w") as f:
            f.write(
                f"scale={scale}\npad_left={pad_left}\npad_top={pad_top}\n"
                f"orig_h={img.shape[0]}\norig_w={img.shape[1]}\n"
            )

    except Exception as e:
        logging.error(e)
        return 1
    else:
        return 0


if __name__ == "__main__":
    count_ok = 0
    count_ng = 0
    for fname in os.listdir("./"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        if fname.lower().endswith("_out.jpg"):
            continue
        logging.info("processing %s ...", fname)
        ret = process(fname)
        if ret == 0:
            logging.info("process %s successfully.", fname)
            count_ok += 1
        else:
            logging.error("failed to process %s", fname)
            count_ng += 1

    total = count_ok + count_ng
    if count_ng == 0:
        logging.info(
            "%s images in total, %s images processed successfully", total, count_ok
        )
    else:
        logging.error("%s images in total, %s ok, %s failed", total, count_ok, count_ng)
