# ============================================================================
# PyTorch 通路测试脚本 - rope_optimized
# ============================================================================
#
# 测试 torch.ops.npu.rope_optimized(x, cos, sin) 与 golden 的精度对比

import sys
import os
import math

import torch
import torch_npu
import numpy as np

from golden import compute_golden

# 算子配置
SO_NAME = "librope_optimized_ops.so"
OP_NAME = "rope_optimized"


def gen_rope_cos_sin(S, D, dtype):
    """生成标准 RoPE 的 cos/sin 表"""
    position = torch.arange(S, dtype=torch.float32).unsqueeze(1)  # [S, 1]
    div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
    angles = position * div_term  # [S, D/2]

    cos = torch.zeros(S, D, dtype=dtype)
    sin = torch.zeros(S, D, dtype=dtype)
    # halves 布局：后半维复制前半维的取值（与 kernel 的 rotate_half 配套，见正文 11.1.1）
    cos[:, : D // 2] = torch.cos(angles)
    cos[:, D // 2 :] = torch.cos(angles)
    sin[:, : D // 2] = torch.sin(angles)
    sin[:, D // 2 :] = torch.sin(angles)
    return cos, sin


def run_test(name, B, S, H, D, dtype):
    """运行单个测试用例，返回 (name, passed, max_diff)"""
    torch.manual_seed(42)

    dt_str = "FP32" if dtype == torch.float32 else "FP16"
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-3

    x = torch.randn(B, S, H, D, dtype=dtype)
    cos, sin = gen_rope_cos_sin(S, D, dtype)

    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x.npu(), cos.npu(), sin.npu())

    golden = compute_golden(x, cos, sin).npu()
    max_diff = torch.max(torch.abs(y - golden)).item()
    passed = torch.allclose(y.cpu(), golden.cpu(), atol=atol, rtol=rtol)

    full_name = f"{name} [{B},{S},{H},{D}] {dt_str}"
    return full_name, passed, max_diff


def main():
    # 从脚本位置推断 build 目录（脚本在 scripts/ 下，build 在上级目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    so_path = os.path.join(project_root, "build", SO_NAME)

    # 如果 project_root/build 不存在，尝试从当前目录推断
    if not os.path.exists(so_path):
        so_path = os.path.join("build", SO_NAME)

    # 如果当前目录已经是 build，直接查找
    if not os.path.exists(so_path):
        so_path = SO_NAME

    if not os.path.exists(so_path):
        print(f"ERROR: {SO_NAME} not found. Run 'cmake .. && make' first.")
        print(f"  Searched: {os.path.join(project_root, 'build', SO_NAME)}, build/{SO_NAME}, ./{SO_NAME}")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # === Level 0: 小规模基础验证 ===
    results.append(run_test("L0 basic", 2, 8, 2, 64, torch.float32))

    # === Level 1: 典型场景 ===
    results.append(run_test("L1 typical", 2, 128, 4, 64, torch.float32))
    results.append(run_test("L1 typical D128", 2, 128, 4, 128, torch.float32))

    # === Level 2: 边界情况 ===
    results.append(run_test("L2 minimal", 1, 1, 1, 64, torch.float32))
    results.append(run_test("L2 zeros", 1, 8, 1, 64, torch.float32))  # cos/sin 不会为 0

    # === FP16 测试 ===
    results.append(run_test("L0 basic FP16", 2, 8, 2, 64, torch.float16))
    results.append(run_test("L1 typical FP16", 2, 128, 4, 64, torch.float16))
    results.append(run_test("L1 typical D128 FP16", 2, 128, 4, 128, torch.float16))

    # === 大规模测试 ===
    results.append(run_test("L2 large FP32", 4, 2048, 8, 64, torch.float32))
    results.append(run_test("L2 large FP16", 4, 2048, 8, 64, torch.float16))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, diff in results:
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}: {status} (Max diff={diff:.6e})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
