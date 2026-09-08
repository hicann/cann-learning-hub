# -*- coding: utf-8 -*-
"""
题目 1（⭐ 入门）：获取并打印 CPU 型号和核心数
答案文件
"""
import os
import platform
import subprocess

# 方式1：使用 Python 标准库
print("=== CPU 信息 ===")
print(f"架构    : {platform.machine()}")
print(f"处理器  : {platform.processor()}")
print(f"核心数  : {os.cpu_count()}")

# 方式2：使用 lscpu 获取更详细信息
print()
r = subprocess.run("lscpu", shell=True, capture_output=True, text=True)
for line in r.stdout.split("\n"):
    if any(k in line for k in ["Architecture", "CPU(s):", "Model name", "CPU max MHz"]):
        print(f"  {line}")
