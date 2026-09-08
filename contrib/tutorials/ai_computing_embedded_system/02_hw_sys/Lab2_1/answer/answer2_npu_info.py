# -*- coding: utf-8 -*-
"""
题目 2（⭐ 入门）：获取并打印 NPU 型号和健康状态
答案文件
"""
import subprocess

r = subprocess.run("npu-smi info", shell=True, capture_output=True, text=True)
print("=== NPU 信息 ===")
print(r.stdout)
