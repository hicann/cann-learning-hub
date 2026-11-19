# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static address arithmetic only; does not execute a defective device kernel."""

import logging
from itertools import combinations

LOGGER = logging.getLogger(__name__)


def overlaps(shape, element_bytes, addresses):
    size = shape[0] * shape[1] * element_bytes
    spans = [(a, a + size) for a in addresses]
    overlapping = []
    for (i, left), (j, right) in combinations(enumerate(spans), 2):
        if max(left[0], right[0]) < min(left[1], right[1]):
            overlapping.append((i, j))
    return overlapping


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    before = overlaps((16, 512), 4, [0, 0x2000, 0x4000])
    after = overlaps((16, 512), 4, [0, 0x8000, 0x10000])
    if before != [(0, 1), (0, 2), (1, 2)] or after != []:
        raise AssertionError('unexpected overlap sets')
    LOGGER.info('重叠组（修复前）: %s', before)
    LOGGER.info('重叠组（修复后）: %s', after)
