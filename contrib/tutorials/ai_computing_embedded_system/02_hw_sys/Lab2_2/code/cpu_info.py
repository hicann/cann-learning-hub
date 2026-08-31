"""
系统硬件信息 - CPU 信息
功能：获取 CPU 架构、核心数、主频等详细信息
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/cpu_info.py

包含三种获取方式：
  1. lscpu 命令 - 格式化输出 CPU 信息
  2. os.cpu_count() / multiprocessing.cpu_count() / nproc - 获取核心数
  3. /proc/cpuinfo - 原始 CPU 详细信息
"""

import os
import subprocess
import multiprocessing

print("=" * 60)
print("CPU 信息 (lscpu)")
print("=" * 60)
result = subprocess.run(['lscpu'], capture_output=True, text=True)
print(result.stdout)

print("=" * 60)
print("CPU 核心数获取方式对比")
print("=" * 60)
print(f"  os.cpu_count()              = {os.cpu_count()}")
print(f"  multiprocessing.cpu_count() = {multiprocessing.cpu_count()}")
result = subprocess.run(['nproc'], capture_output=True, text=True)
print(f"  nproc                       = {result.stdout.strip()}")
result = subprocess.run(['bash', '-c', 'grep -c processor /proc/cpuinfo'], capture_output=True, text=True)
print(f"  /proc/cpuinfo 核心数        = {result.stdout.strip()}")

print()
print("=" * 60)
print("/proc/cpuinfo（前30行）")
print("=" * 60)
result = subprocess.run(['bash', '-c', 'cat /proc/cpuinfo | head -30'], capture_output=True, text=True)
print(result.stdout)
