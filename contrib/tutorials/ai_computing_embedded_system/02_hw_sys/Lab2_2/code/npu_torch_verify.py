"""
PyTorch + torch_npu NPU 可用性验证
功能：验证 NPU 是否可以正常用于 AI 计算（矩阵乘法）
适用平台：香橙派 AIpro（昇腾 310B4）/ Atlas 910B3
运行方式：python3 npu_torch_verify.py

依赖包：torch, torch_npu
若未安装，建议执行：
    pip install torch
    pip install torch_npu
"""

import ctypes
import ctypes.util
import os

# 预加载 libgomp，避免 "cannot allocate memory in static TLS block" 错误
# 原理：PyTorch 依赖的 OpenMP 库需要在进程启动时即加载到静态 TLS 块中
try:
    _gomp = ctypes.util.find_library('gomp')
    if _gomp:
        ctypes.CDLL(_gomp, mode=ctypes.RTLD_GLOBAL)
    else:
        for _p in ['/usr/lib/x86_64-linux-gnu/libgomp.so.1',
                    '/usr/lib/aarch64-linux-gnu/libgomp.so.1',
                    '/lib64/libgomp.so.1',
                    '/usr/lib64/libgomp.so.1']:
            if os.path.exists(_p):
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
                break
except Exception:
    pass

import torch
import torch_npu

print(f'NPU 可用: {torch.npu.is_available()}')
print(f'NPU 卡数: {torch.npu.device_count()}')
print(f'NPU 名称: {torch.npu.get_device_name(0)}')
print(f'当前 NPU 设备: {torch.npu.current_device()}')

# 简单矩阵乘法验证
A = torch.randn(100, 100, device='npu:0')
B = torch.randn(100, 100, device='npu:0')
C = torch.mm(A, B)
torch.npu.synchronize()
print(f'矩阵乘法验证通过，结果形状: {C.shape}，设备: {C.device}')
