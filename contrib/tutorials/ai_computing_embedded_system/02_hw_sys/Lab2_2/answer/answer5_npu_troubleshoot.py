"""
思考题 5 参考答案：若 NPU 负载始终为 0% 而 CPU 满载，应如何排查？

排查步骤：
    1. 检查环境变量是否正确配置
       - 执行 echo $ASCEND_TOOLKIT_HOME 确认路径
       - 执行 source set_env.sh 重新配置

    2. 检查模型是否成功加载到 NPU
       - 确认代码中使用了 .npu() 将模型和数据搬到 NPU
       - 检查 torch.npu.is_available() 返回 True

    3. 检查是否存在 CPU fallback
       - 某些算子可能不支持 NPU，会自动回退到 CPU 执行
       - 查看日志中是否有 fallback 警告信息

    4. 使用 npu-smi info 监控 NPU 状态
       - 观察 AICore(%) 是否有变化
       - 观察 Memory-Usage 是否有显存占用

    5. 检查 CANN 版本与芯片型号是否匹配
       - 确认 SOC_VERSION 设置正确
"""

import subprocess

print("思考题 5 答案：NPU 负载为 0% 的排查步骤")
print()

steps = [
    "1. 检查环境变量: echo $ASCEND_TOOLKIT_HOME",
    "2. 检查 NPU 可用性: python3 -c \"import torch_npu; print(torch.npu.is_available())\"",
    "3. 检查模型设备: 确认使用了 .npu() 搬运模型和数据",
    "4. 监控 NPU 状态: npu-smi info",
    "5. 检查 CANN 版本与芯片型号匹配",
]

for step in steps:
    print(f"  {step}")

print()
print("当前 NPU 状态：")
try:
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=5)
    print(result.stdout)
except Exception as e:
    print(f"  无法获取 NPU 信息: {e}")
