"""
系统硬件信息 - NPU（昇腾 AI 处理器）信息
功能：使用 npu-smi 查看 NPU 型号、状态、利用率、显存等信息
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/npu_info.py

常用命令：
  npu-smi info            - NPU 概览信息
  npu-smi info -l         - NPU 设备列表/拓扑信息
  npu-smi info -t board   - 板卡详细信息
  npu-smi info -t usages  - 资源使用情况
"""

import subprocess

print("=" * 60)
print("[1] NPU 概览信息 (npu-smi info)")
print("=" * 60)
result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=10)
print(result.stdout)

print("=" * 60)
print("[2] NPU 设备列表 (npu-smi info -l)")
print("=" * 60)
result = subprocess.run(['npu-smi', 'info', '-l'], capture_output=True, text=True, timeout=10)
print(result.stdout)

# 自动解析 Card ID，获取板卡详细信息和资源使用情况
result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
card_ids = set()
for line in result.stdout.split("\n"):
    parts = line.split()
    if len(parts) >= 2 and parts[0].isdigit():
        card_ids.add(int(parts[0]))

for cid in sorted(card_ids):
    print("=" * 60)
    print(f"[3] Card ID {cid} 板卡信息 (npu-smi info -t board -i {cid})")
    print("=" * 60)
    r = subprocess.run(['npu-smi', 'info', '-t', 'board', '-i', str(cid)],
                       capture_output=True, text=True, timeout=10)
    if r.stdout.strip():
        print(r.stdout)
    if r.stderr.strip():
        print(f"(board 信息不可用: {r.stderr.strip()})")

    print("=" * 60)
    print(f"[4] Card ID {cid} 资源使用 (npu-smi info -t usages -i {cid})")
    print("=" * 60)
    r2 = subprocess.run(['npu-smi', 'info', '-t', 'usages', '-i', str(cid)],
                        capture_output=True, text=True, timeout=10)
    if r2.stdout.strip():
        print(r2.stdout)
    if r2.stderr.strip():
        print(f"(usages 信息不可用: {r2.stderr.strip()})")
