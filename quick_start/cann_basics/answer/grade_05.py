"""grade_05.py — CANN 基础课程总结实践题批改脚本

三道实践题的参考答案与验证逻辑：
  第 1 题（基础）: z = x² + y，验证所有元素等于 7.0
  第 2 题（进阶）: CPU vs NPU 矩阵乘法性能对比
  第 3 题（挑战）: Batch 矩阵乘法 vs 循环单次矩阵乘法性能对比
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
        sizes = [128, 512, 2048]
        all_ok = True
        for N in sizes:
            a_cpu = torch.randn(N, N, dtype=torch.float32)
            b_cpu = torch.randn(N, N, dtype=torch.float32)

            # CPU
            t0 = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            t_cpu = time.time() - t0

            # NPU
            a_npu = a_cpu.npu()
            b_npu = b_cpu.npu()
            torch.npu.synchronize()
            t0 = time.time()
            c_npu = torch.matmul(a_npu, b_npu)
            torch.npu.synchronize()
            t_npu = time.time() - t0

            consistent = torch.allclose(c_cpu, c_npu.cpu(), atol=1e-1)
            speedup = t_cpu / t_npu if t_npu > 0 else float("inf")
            tag = "✓" if consistent else "✗"
            print(f"  N={N:>4}: CPU={t_cpu*1000:>8.2f}ms  NPU={t_npu*1000:>8.2f}ms  "
                  f"加速比={speedup:>6.2f}x  一致={tag}")
            if not consistent:
                all_ok = False
        print(f"  结果: {'✓ 全部规模结果一致' if all_ok else '✗ 部分规模结果不一致'}")
        return all_ok
    except Exception as e:
        print(f"  ✗ 验证异常: {e}")
        return False


def _check_practice_3(g):
    """第 3 题：Batch 矩阵乘法 vs 循环单次矩阵乘法性能对比"""
    print("\n--- 第 3 题（挑战）：Batch 矩阵乘法 vs 循环单次矩阵乘法 ---")
    try:
        batch_sizes = [8, 32, 128]
        M, K, N = 256, 256, 256
        reps = 20
        all_ok = True

        for B in batch_sizes:
            a = torch.randn(B, M, K, dtype=torch.float32).npu()
            b = torch.randn(B, K, N, dtype=torch.float32).npu()

            # warmup
            for _ in range(3):
                _ = torch.bmm(a, b)
                for i in range(B):
                    _ = torch.matmul(a[i], b[i])
            torch.npu.synchronize()

            # bmm
            torch.npu.synchronize()
            t0 = time.time()
            for _ in range(reps):
                c_bmm = torch.bmm(a, b)
            torch.npu.synchronize()
            t_bmm = (time.time() - t0) / reps * 1000

            # loop
            torch.npu.synchronize()
            t0 = time.time()
            for _ in range(reps):
                c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
            torch.npu.synchronize()
            t_loop = (time.time() - t0) / reps * 1000

            consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)
            speedup = t_loop / t_bmm if t_bmm > 0 else float("inf")
            tag = "✓" if consistent else "✗"
            print(f"  B={B:>4}: bmm={t_bmm:>8.2f}ms  loop={t_loop:>8.2f}ms  "
                  f"加速比={speedup:>6.2f}x  一致={tag}")

            if B >= 32 and speedup < 2.0:
                print(f"    ✗ 加速不明显（{speedup:.2f}x < 2.0x）")
                all_ok = False
            if not consistent:
                all_ok = False

        print(f"  结果: {'✓ bmm 性能优势符合预期' if all_ok else '✗ 部分指标未达预期'}")
        return all_ok
    except Exception as e:
        print(f"  ✗ 验证异常: {e}")
        return False


def _print_reference_answers():
    print("=" * 60)
    print("  参考答案")
    print("=" * 60)
    print("""
# ── 第 1 题 ──────────────────────────────────────
x = torch.full((1000,), 2.0, dtype=torch.float32)
y = torch.full((1000,), 3.0, dtype=torch.float32)
x_npu = x.npu()
y_npu = y.npu()
z_npu = x_npu * x_npu + y_npu   # 或 torch.pow(x_npu, 2) + y_npu
z = z_npu.cpu()

# ── 第 2 题 ──────────────────────────────────────
for N in sizes:
    a_cpu = torch.randn(N, N, dtype=torch.float32)
    b_cpu = torch.randn(N, N, dtype=torch.float32)
    # CPU
    t0 = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    t_cpu = time.time() - t0
    # NPU
    a_npu, b_npu = a_cpu.npu(), b_cpu.npu()
    torch.npu.synchronize()
    t0 = time.time()
    c_npu = torch.matmul(a_npu, b_npu)
    torch.npu.synchronize()
    t_npu = time.time() - t0
    consistent = torch.allclose(c_cpu, c_npu.cpu(), atol=1e-1)

# ── 第 3 题 ──────────────────────────────────────
batch_sizes = [8, 32, 128]
M, K, N = 256, 256, 256
reps = 20
for B in batch_sizes:
    a = torch.randn(B, M, K, dtype=torch.float32).npu()
    b = torch.randn(B, K, N, dtype=torch.float32).npu()
    # warmup
    for _ in range(3): torch.bmm(a, b)
    torch.npu.synchronize()
    # bmm
    t0 = time.time()
    for _ in range(reps): c_bmm = torch.bmm(a, b)
    torch.npu.synchronize(); t_bmm = (time.time()-t0)/reps*1000
    # loop
    t0 = time.time()
    for _ in range(reps):
        c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
    torch.npu.synchronize(); t_loop = (time.time()-t0)/reps*1000
    consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)
    print(f"B={B}: bmm={t_bmm:.2f}ms loop={t_loop:.2f}ms 加速={t_loop/t_bmm:.2f}x 一致={consistent}")
""")
