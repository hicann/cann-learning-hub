"""
系统硬件信息 - CPU 与 NPU 负载监控
功能：实时监控 CPU 利用率和 NPU 资源使用情况
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/load_monitor.py

监控内容：
  1. CPU 实时利用率 (top 快照)
  2. NPU 详细资源使用情况 (npu-smi info -t usages)
"""

import subprocess

print("=" * 60)
print("CPU 负载监控 (top 快照)")
print("=" * 60)
result = subprocess.run(['bash', '-c', 'top -bn1 | head -20'], capture_output=True, text=True)
print(result.stdout)

# NPU 负载监控
print("=" * 60)
print("NPU 负载监控 (npu-smi info -t usages)")
print("=" * 60)

result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
card_ids = set()
for line in result.stdout.split("\n"):
    parts = line.split()
    if len(parts) >= 2 and parts[0].isdigit():
        card_ids.add(int(parts[0]))

if card_ids:
    for cid in sorted(card_ids):
        print(f"--- NPU Card {cid} 资源使用详情 ---")
        r = subprocess.run(['npu-smi', 'info', '-t', 'usages', '-i', str(cid)],
                           capture_output=True, text=True, timeout=10)
        if r.stdout.strip():
            print(r.stdout)
        if r.stderr.strip():
            print(f"(详细信息不可用: {r.stderr.strip()})")
else:
    print("未检测到 NPU 设备")
