"""
Python 版本信息
功能：获取 Python 解释器的版本号和路径，确认当前使用的 Python 环境
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/python_info.py
"""

import sys
import platform

print("=" * 60)
print("Python 环境信息")
print("=" * 60)
print(f"  Python 版本   : {sys.version}")
print(f"  Python 路径   : {sys.executable}")
print(f"  平台标识      : {sys.platform}")
print(f"  机器架构      : {platform.machine()}")
print(f"  处理器型号    : {platform.processor()}")
print(f"  操作系统      : {platform.platform()}")
print(f"  字节序        : {sys.byteorder}")
