"""
操作系统信息
功能：获取操作系统发行版、内核版本、运行时间等信息
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/os_info.py

获取内容：
  1. /etc/os-release - OS 发行版信息
  2. uname -a       - 内核版本、主机名、架构
  3. /proc/version  - 内核编译信息
  4. uptime         - 系统运行时间和负载平均值
  5. who -b         - 系统启动时间
"""

import subprocess

print("=" * 60)
print("操作系统信息")
print("=" * 60)

print("\n--- /etc/os-release ---")
result = subprocess.run(['cat', '/etc/os-release'], capture_output=True, text=True)
print(result.stdout)

print("--- uname -a ---")
result = subprocess.run(['uname', '-a'], capture_output=True, text=True)
print(result.stdout)

print("--- 内核版本 (/proc/version) ---")
result = subprocess.run(['cat', '/proc/version'], capture_output=True, text=True)
print(result.stdout)

print("--- 系统运行时间和负载平均值 (uptime) ---")
result = subprocess.run(['uptime'], capture_output=True, text=True)
print(result.stdout)

print("--- 系统启动时间 (who -b) ---")
result = subprocess.run(['who', '-b'], capture_output=True, text=True)
print(result.stdout)
