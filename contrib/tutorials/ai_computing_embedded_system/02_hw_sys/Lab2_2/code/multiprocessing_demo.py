"""
多核 CPU 并行程序
功能：对比串行和并行执行的加速效果，演示 multiprocessing 多进程并行
适用平台：香橙派 AIpro（昇腾 310B4）
运行方式：python3 code/multiprocessing_demo.py

包含两个示例：
  A. 最简单的多进程示例 - 启动多个进程同时打印消息
  B. 串行 vs 并行对比  - 对比执行时间和加速比
"""

import multiprocessing
import os
import time

# ===== A. 最简单的多进程示例 =====
print("=" * 60)
print("A. 最简单的多进程示例")
print("=" * 60)

num_cores = os.cpu_count()
print(f"本机 CPU 核心数: {num_cores}")


def hello_worker(worker_id):
    return f"Worker {worker_id}: Hello! (来自进程)"


num = min(4, num_cores)
print(f"\n启动 {num} 个进程：")
with multiprocessing.Pool(num) as pool:
    results = pool.map(hello_worker, range(num))

for r in results:
    print(f"  {r}")
print("所有进程完成！")

# ===== B. 串行 vs 并行对比 =====
print()
print("=" * 60)
print("B. 串行 vs 并行对比")
print("=" * 60)


def simple_sum(n):
    return sum(range(n))


N = 5_000_000
num = min(4, multiprocessing.cpu_count())

# 串行：一个核心做 num 次
start = time.time()
results_serial = [simple_sum(N) for _ in range(num)]
t_serial = time.time() - start

# 并行：num 个核心各做 1 次
start = time.time()
with multiprocessing.Pool(num) as pool:
    results_parallel = pool.map(simple_sum, [N] * num)
t_parallel = time.time() - start

print(f"任务: 计算 0+1+2+...+{N-1}，共 {num} 次")
print(f"串行 (1核心做{num}次): {t_serial:.3f} 秒")
print(f"并行 ({num}核心各做1次): {t_parallel:.3f} 秒")
print(f"加速比: {t_serial / t_parallel:.1f}x")
print(f"结果一致: {results_serial == results_parallel}")
