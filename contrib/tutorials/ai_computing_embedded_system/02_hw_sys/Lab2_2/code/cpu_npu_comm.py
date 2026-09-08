"""
CPU 与 NPU 通信
功能：演示张量在 CPU 和 NPU 之间搬运的基本操作
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/cpu_npu_comm.py

依赖包：torch, torch_npu
若未安装，建议执行：
    pip install torch
    pip install torch_npu

演示流程：在 CPU 上创建数据 -> 搬到 NPU -> 搬回 CPU
"""

import ctypes
import ctypes.util
import os

# 预加载 libgomp，避免 "cannot allocate memory in static TLS block" 错误
try:
    _gomp = ctypes.util.find_library('gomp')
    if _gomp:
        ctypes.CDLL(_gomp, mode=ctypes.RTLD_GLOBAL)
    else:
        for _p in ['/usr/lib/aarch64-linux-gnu/libgomp.so.1',
                    '/lib64/libgomp.so.1',
                    '/usr/lib64/libgomp.so.1']:
            if os.path.exists(_p):
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
                break
except Exception:
    pass

import torch
import torch_npu

print(f"NPU 是否可用: {torch.npu.is_available()}")
print()

# 第1步：在 CPU 上创建一个张量
x = torch.tensor([1.0, 2.0, 3.0])
print(f"第1步 - CPU 上的张量: {x}")
print(f"        所在设备: {x.device}")
print()

# 第2步：把数据从 CPU 搬到 NPU
x_npu = x.npu()
print(f"第2步 - 已搬到 NPU: {x_npu}")
print(f"        所在设备: {x_npu.device}")
print()

# 第3步：把数据从 NPU 搬回 CPU
x_back = x_npu.cpu()
print(f"第3步 - 搬回 CPU: {x_back}")
print(f"        所在设备: {x_back.device}")
print()

print("小结: 数据成功从 CPU -> NPU -> CPU 搬了一个来回！")
