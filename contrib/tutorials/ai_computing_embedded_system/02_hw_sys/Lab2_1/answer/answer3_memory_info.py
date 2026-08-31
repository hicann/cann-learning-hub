# -*- coding: utf-8 -*-
"""
题目 3（⭐⭐ 基硷）：获取内存总量与使用率
答案文件
"""
import subprocess

# 方式1：使用 free 命令
print("=== 内存信息 (free -h) ===")
r = subprocess.run("free -h", shell=True, capture_output=True, text=True)
print(r.stdout)

# 方式2：解析 /proc/meminfo
print("=== /proc/meminfo 解析 ===")
r2 = subprocess.run("cat /proc/meminfo", shell=True, capture_output=True, text=True)
meminfo = {}
for line in r2.stdout.split("\n"):
    parts = line.split(":")
    if len(parts) == 2:
        key = parts[0].strip()
        val = parts[1].strip().split()[0]
        meminfo[key] = int(val)

total = meminfo.get("MemTotal", 0) / 1024  # MB -> GB
free = meminfo.get("MemFree", 0) / 1024
available = meminfo.get("MemAvailable", 0) / 1024
used = total - available
usage_pct = used / total * 100 if total > 0 else 0

print(f"内存总量    : {total:.2f} GB")
print(f"已使用      : {used:.2f} GB")
print(f"可用        : {available:.2f} GB")
print(f"使用率      : {usage_pct:.1f}%")

swap_total = meminfo.get("SwapTotal", 0) / 1024
swap_free = meminfo.get("SwapFree", 0) / 1024
print(f"交换分区总量: {swap_total:.2f} GB")
print(f"交换分区空闲: {swap_free:.2f} GB")
