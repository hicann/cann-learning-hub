"""
系统硬件信息 - 磁盘信息
功能：使用 df 和 lsblk 查看系统磁盘空间和块设备信息
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/disk_info.py
"""

import subprocess

print("=" * 60)
print("磁盘使用情况 (df -h)")
print("=" * 60)
result = subprocess.run(['df', '-h'], capture_output=True, text=True)
print(result.stdout)

print("=" * 60)
print("块设备信息 (lsblk)")
print("=" * 60)
result = subprocess.run(['lsblk'], capture_output=True, text=True)
print(result.stdout)
