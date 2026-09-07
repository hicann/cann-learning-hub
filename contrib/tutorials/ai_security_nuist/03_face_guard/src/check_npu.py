# -*- coding: utf-8 -*-
"""
检查 NPU / PyTorch 环境。

优先尝试 Ascend NPU (torch_npu)，并做一次小矩阵乘法验证。
若无 NPU，则报告 CUDA/CPU，便于在本地或迁移途中排查。

迁移到 NPU 服务器后，先执行：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
再运行本脚本。理想输出：
    NPU available: True
    NPU count: 1
    Tensor device: npu:0
    NPU calculation passed
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    try:
        import torch
        print("Python:", sys.version.split()[0])
        print("PyTorch:", torch.__version__)
    except Exception as exc:
        print("PyTorch unavailable:", exc)
        return 2

    # ---- NPU ----
    try:
        import torch_npu
        print("torch_npu:", torch_npu.__version__)
        print("NPU available:", torch.npu.is_available())
        print("NPU count:", torch.npu.device_count())
        if torch.npu.is_available():
            torch.npu.set_device(0)
            x = torch.randn(4, 4).npu()
            y = x @ x
            print("Tensor device:", y.device)
            print("NPU calculation passed")
            return 0
        print("torch_npu imported but no NPU device visible")
    except Exception as exc:
        print("torch_npu unavailable:", repr(exc))

    # ---- CUDA fallback ----
    if torch.cuda.is_available():
        print("CUDA available, count:", torch.cuda.device_count())
        x = torch.randn(4, 4).cuda()
        y = x @ x
        print("Tensor device:", y.device)
        print("CUDA calculation passed")
        return 0

    # ---- CPU ----
    print("Running on CPU only (no NPU/CUDA).")
    x = torch.randn(4, 4)
    y = x @ x
    print("CPU calculation passed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
