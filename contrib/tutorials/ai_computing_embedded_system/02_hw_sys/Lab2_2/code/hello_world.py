"""
Hello World 程序
功能：最简单的 Python 程序，验证环境可正常运行
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/hello_world.py
"""

import sys
import platform

print("Hello World!")
print("Hello Ascend NPU!")
print("Welcome to Orange Pi AIpro!")

print(f"\n--- 环境信息 ---")
print(f"Python: {sys.version.split()[0]}")
print(f"平台  : {platform.machine()}")
print(f"OS    : {platform.platform()}")
