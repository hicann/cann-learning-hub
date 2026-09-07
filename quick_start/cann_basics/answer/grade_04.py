"""grade_04.py — CANN 基础 NPU 实践题批改脚本

四道实践题的参考答案与验证逻辑：
  第 1 题（基础）: z = x² + y，验证所有元素等于 7.0
  第 2 题（进阶）: CPU vs NPU 矩阵乘法性能对比
  第 3 题（挑战）: Batch 矩阵乘法 vs 循环单次矩阵乘法性能对比
  第 4 题（挑战）: NPU 图像高斯模糊（卷积运算）
"""

import time
import torch


def grade(user_globals):
    print("=" * 60)
    print("  CANN 基础课程实践题 — 批改结果")
    print("=" * 60)

    results = []
    results.append(_check_practice_1(user_globals))
    results.append(_check_practice_2(user_globals))
    results.append(_check_practice_3(user_globals))
    results.append(_check_practice_4(user_globals))

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    for i, ok in enumerate(results):
        tag = "✓ 通过" if ok else "✗ 未通过（或未完成）"
        print(f"  第 {i+1} 题: {tag}")
    print("=" * 60)
    if passed == total:
        print(f"🎉 全部 {total} 道题通过！")
    else:
        print(f"得分：{passed}/{total}")
    print()
    _print_reference_answers()


def _check_practice_1(g):
    """第 1 题：z = x² + y，验证所有元素等于 7.0"""
    print("\n--- 第 1 题（基础）：z = x² + y ---")
    try:
        z = g.get("z")
        if z is None:
            print("  ✗ 未找到变量 z，请完成代码后运行。")
            return False
        expected = torch.full((1000,), 7.0)
        ok = torch.allclose(z, expected)
        z_npu = g.get("z_npu")
        on_npu = z_npu is not None and "npu" in str(z_npu.device)
        print(f"  z 形状: {tuple(z.shape)}, dtype: {z.dtype}")
        print(f"  z 前 5 个元素: {z[:5].tolist()}")
        print(f"  计算在 NPU 上执行: {'✓' if on_npu else '✗'}")
        print(f"  所有元素等于 7.0: {'✓' if ok else '✗'}")
        return ok and on_npu
    except Exception as e:
        print(f"  ✗ 验证异常: {e}")
        return False


def _check_practice_2(g):
    """第 2 题：CPU vs NPU 矩阵乘法性能对比"""
    print("\n--- 第 2 题（进阶）：CPU vs NPU 矩阵乘法性能对比 ---")
    try:
        c_cpu = g.get("c_cpu")
        c_npu = g.get("c_npu")
        if c_cpu is None or c_npu is None:
            print("  ✗ 未找到变量 c_cpu / c_npu，请完成代码后运行。")
            return False

        # 检查最后一次迭代（N=2048）的结果
        consistent = torch.allclose(c_cpu, c_npu.cpu(), atol=1e-1)
        print(f"  c_cpu 形状: {tuple(c_cpu.shape)}")
        print(f"  c_npu 设备: {c_npu.device}")
        print(f"  结果一致: {'✓' if consistent else '✗'}")

        # 也自己跑一遍验证概念
        for N in [128, 512, 2048]:
            a_cpu = torch.randn(N, N, dtype=torch.float32)
            b_cpu = torch.randn(N, N, dtype=torch.float32)
            t0 = time.time()
            _ = torch.matmul(a_cpu, b_cpu)
            t_cpu = time.time() - t0
            a_npu, b_npu = a_cpu.npu(), b_cpu.npu()
            torch.npu.synchronize()
            t0 = time.time()
            _ = torch.matmul(a_npu, b_npu)
            torch.npu.synchronize()
            t_npu = time.time() - t0
            print(f"  N={N:>4}: CPU={t_cpu*1000:>8.2f}ms  NPU={t_npu*1000:>8.2f}ms  加速比={t_cpu/t_npu:>6.2f}x")

        return consistent
    except Exception as e:
        print(f"  ✗ 验证异常: {e}")
        return False


def _check_practice_3(g):
    """第 3 题：Batch 矩阵乘法 vs 循环单次矩阵乘法性能对比"""
    print("\n--- 第 3 题（挑战）：Batch 矩阵乘法 vs 循环单次矩阵乘法 ---")
    try:
        c_bmm = g.get("c_bmm")
        c_loop = g.get("c_loop")
        if c_bmm is None or c_loop is None:
            print("  ✗ 未找到变量 c_bmm / c_loop，请完成代码后运行。")
            return False

        # 检查最后一次迭代（B=128）的结果
        consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)
        print(f"  c_bmm 形状: {tuple(c_bmm.shape)}")
        print(f"  c_loop 形状: {tuple(c_loop.shape)}")
        print(f"  结果一致: {'✓' if consistent else '✗'}")

        # 也自己跑一遍验证概念
        B, M, K, N = 128, 256, 256, 256
        a = torch.randn(B, M, K, dtype=torch.float32).npu()
        b = torch.randn(B, K, N, dtype=torch.float32).npu()
        for _ in range(3): torch.bmm(a, b)
        torch.npu.synchronize()
        t0 = time.time()
        for _ in range(20): _ = torch.bmm(a, b)
        torch.npu.synchronize(); t_bmm = (time.time()-t0)/20*1000
        t0 = time.time()
        for _ in range(20): _ = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
        torch.npu.synchronize(); t_loop = (time.time()-t0)/20*1000
        print(f"  B=128: bmm={t_bmm:.2f}ms loop={t_loop:.2f}ms 加速={t_loop/t_bmm:.1f}x")

        return consistent
    except Exception as e:
        print(f"  ✗ 验证异常: {e}")
        return False


def _check_practice_4(g):
    """第 4 题：NPU 图像高斯模糊（卷积）"""
    print("\n--- 第 4 题（挑战）：NPU 图像高斯模糊（卷积） ---")
    try:
        result = g.get("result")
        blur_cpu = g.get("blur_cpu")
        if result is None or blur_cpu is None:
            print("  ✗ 未找到变量 result / blur_cpu，请完成代码后运行。")
            return False

        # 检查输出形状
        print(f"  result 形状: {tuple(result.shape)}")
        if result.shape != (1, 3, 512, 512):
            print(f"  ✗ 输出形状不符合预期 (1, 3, 512, 512)")
            return False

        # 检查 CPU 与 NPU 结果一致性
        diff = (blur_cpu - result).abs().max().item()
        consistent = diff < 1e-3
        print(f"  CPU 与 NPU 最大差异: {diff:.6f}")
        print(f"  结果一致: {'✓' if consistent else '✗'}")

        # 检查是否在 NPU 上执行
        npu_time = g.get("npu_time")
        cpu_time = g.get("cpu_time")
        if npu_time and cpu_time:
            speedup = cpu_time / npu_time
            print(f"  CPU: {cpu_time*1000:.2f} ms, NPU: {npu_time*1000:.2f} ms, 加速比: {speedup:.1f}x")

        return consistent
    except Exception as e:
        print(f"  ✗ 验证异常: {e}")
        return False


def _print_reference_answers():
    print("\n  参考答案已保存在 answer/grade_04.py 中，请直接查看源文件。")


# ── 参考答案（仅供查阅，不在批改时打印） ──────────────────
#
# 第 1 题:
#   x = torch.full((1000,), 2.0, dtype=torch.float32)
#   y = torch.full((1000,), 3.0, dtype=torch.float32)
#   z_npu = x.npu() * x.npu() + y.npu()
#   z = z_npu.cpu()
#
# 第 2 题:
#   for N in [128, 512, 2048]:
#       a_cpu = torch.randn(N, N, dtype=torch.float32)
#       b_cpu = torch.randn(N, N, dtype=torch.float32)
#       c_cpu = torch.matmul(a_cpu, b_cpu)
#       a_npu, b_npu = a_cpu.npu(), b_cpu.npu()
#       torch.npu.synchronize()
#       c_npu = torch.matmul(a_npu, b_npu)
#       torch.npu.synchronize()
#       consistent = torch.allclose(c_cpu, c_npu.cpu(), atol=1e-1)
#
# 第 3 题:
#   for B in [8,2, 32, 128]:
#       a = torch.randn(B, 256, 256, dtype=torch.float32).npu()
#       b = torch.randn(B, 256, 256, dtype=torch.float32).npu()
#       c_bmm = torch.bmm(a, b)
#       c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
#       consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)
#
# 第 4 题:
#   blur_cpu = F.conv2d(img_tensor, weight, padding=2, groups=3)
#   img_npu = img_tensor.npu()
#   weight_npu = weight.npu()
#   torch.npu.synchronize()
#   blur_npu = F.conv2d(img_npu, weight_npu, padding=2, groups=3)
#   torch.npu.synchronize()
#   result = blur_npu.cpu()
