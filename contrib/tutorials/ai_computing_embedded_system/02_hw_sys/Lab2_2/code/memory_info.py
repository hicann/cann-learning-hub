"""
系统硬件信息 - 内存信息
功能：使用 free 和 /proc/meminfo 查看系统内存使用情况
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/memory_info.py
"""

import subprocess

print("=" * 60)
print("内存信息 (free -h)")
print("=" * 60)
result = subprocess.run(['free', '-h'], capture_output=True, text=True)
print(result.stdout)

print("=" * 60)
print("/proc/meminfo（前10行）")
print("=" * 60)
result = subprocess.run(['bash', '-c', 'cat /proc/meminfo | head -10'], capture_output=True, text=True)
print(result.stdout)
