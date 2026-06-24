import os
import csv
import torch
import torch_npu
import custom_ops
from pathlib import Path

torch.npu.config.allow_internal_format = True
torch.npu.config.allow_hf32 = False
torch.npu.set_compile_mode(jit_compile=False)

# ============================================================
# 测试配置: 多组 (M, K, N) 规格
# ============================================================
TEST_SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 6144),
    (1, 4096, 24576),
    (1, 12288, 4096),
    (50, 4096, 4096),
    (50, 4096, 6144),
    (50, 4096, 24576),
    (50, 12288, 4096),
    (4096, 4096, 4096),
    (4096, 4096, 6144),
    (4096, 4096, 24576),
    (4096, 12288, 4096),
]

# ============================================================
# Profiler 配置
# ============================================================
PROF_OUTPUT_DIR = str(TUTORIAL_DIR / "prof_output")
WARMUP_ITERS = 0      # warmup 步数, 不采集数据


def check_close(output, ref, dtype):
    """统一的精度比对函数，根据输出 dtype 使用不同的 tolerance"""
    if dtype == torch.int32:
        # 整数运算应精确匹配，tolerance 为 0
        rtol, atol = 0, 0
    else:
        # BF16 允许浮点舍入误差
        rtol, atol = 0.01, 0.01

    output_f = output.to(torch.float32)
    ref_f = ref.to(torch.float32)
    passed = torch.allclose(output_f, ref_f, rtol=rtol, atol=atol)

    label = 'INT32' if dtype == torch.int32 else 'BF16'
    print(f"  {label} allclose验证: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"  output[0,0:5] = {output[0, 0:5].tolist()}")
        print(f"  ref[0,0:5]    = {ref[0, 0:5].tolist()}")

    return passed


def test_int32(M, K, N):
    """无 pertoken_scale, 纯INT8 matmul, INT32 输出"""
    x1_cpu = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    x2_cpu = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    scale_cpu = torch.randn([N], dtype=torch.float32)

    x1_npu = x1_cpu.npu()
    x2_npu = x2_cpu.npu()
    x2_npu = torch_npu.npu_format_cast(x2_npu, 29)
    scale_npu = scale_cpu.npu()

    output_npu = torch.ops.custom.qmm_custom(x1_npu, x2_npu, scale_npu, pertoken_scale=None, dtype=0)
    output_cpu = output_npu.cpu()

    # float64 matmul 走 BLAS 优化, 比 int32 朴素乘加快很多;
    # float64 精度足够精确表示 int32 范围内的整数结果
    ref_output = torch.matmul(x1_cpu.to(torch.float64), x2_cpu.to(torch.float64)).round().to(torch.int32)

    assert output_cpu.shape == ref_output.shape, f"shape不一致: {output_cpu.shape} vs {ref_output.shape}"
    assert output_cpu.dtype == torch.int32, f"输出dtype不正确: {output_cpu.dtype}"

    return check_close(output_cpu, ref_output, torch.int32)


def test_bf16(M, K, N):
    """有 pertoken_scale, INT8 matmul + dequant, BF16 输出"""
    x1_cpu = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    x2_cpu = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    scale_cpu = torch.randn([N], dtype=torch.float32)
    pertoken_scale_cpu = torch.randn([M], dtype=torch.float32)

    x1_npu = x1_cpu.npu()
    x2_npu = x2_cpu.npu()
    x2_npu = torch_npu.npu_format_cast(x2_npu, 29)
    scale_npu = scale_cpu.npu()
    pertoken_scale_npu = pertoken_scale_cpu.npu()

    output_npu = torch.ops.custom.qmm_custom(x1_npu, x2_npu, scale_npu,
                                        pertoken_scale=pertoken_scale_npu, dtype=1)
    output_cpu = output_npu.cpu()

    int32_result = torch.matmul(x1_cpu.to(torch.float64), x2_cpu.to(torch.float64)).round().to(torch.int32)
    ref_bf16 = (int32_result.to(torch.float32) * scale_cpu.unsqueeze(0) * pertoken_scale_cpu.unsqueeze(1)).to(torch.bfloat16)

    assert output_cpu.shape == ref_bf16.shape, f"shape不一致: {output_cpu.shape} vs {ref_bf16.shape}"
    assert output_cpu.dtype == torch.bfloat16, f"输出dtype不正确: {output_cpu.dtype}"

    return check_close(output_cpu, ref_bf16, torch.bfloat16)


def parse_kernel_details(csv_path):
    """解析 kernel_details.csv, 提取 QmmCustom 每次执行的 Duration
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            if name != "QmmCustom":
                continue

            # 解析 Input Shapes: 格式如 "1,4096;4096,4096;4096;"
            raw_shapes = row.get("Input Shapes", "").strip('"')
            parts = [p.strip() for p in raw_shapes.split(";")]
            x1_shape = parts[0].split(",")  # (M, K)
            x2_shape = parts[1].split(",")  # (K, N)
            M = int(x1_shape[0])
            K = int(x1_shape[1])
            N = int(x2_shape[1])

            # 从 Output Data Types 判断 dtype
            out_dtype_raw = row.get("Output Data Types", "").strip('"')
            dtype_tag = "INT32" if "INT32" in out_dtype_raw else "BF16"

            # Duration(us)
            duration_us = float(row.get("Duration(us)", "0") or "0")

            rows.append({
                "M": M, "K": K, "N": N,
                "dtype": dtype_tag,
                "duration_us": duration_us,
            })
    return rows


def find_kernel_details_csv(prof_dir):
    prof_path = Path(prof_dir)
    if not prof_path.exists():
        return None
    # 找出所有包含 ASCEND_PROFILER_OUTPUT/kernel_details.csv 的子目录
    candidates = []
    for subdir in sorted(prof_path.iterdir(), key=lambda d: d.name, reverse=True):
        csv_path = subdir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        if csv_path.exists():
            candidates.append(csv_path)
            break  # sorted reverse=True, 第一个就是最新的
    return str(candidates[0]) if candidates else None


# ============================================================
# 主流程: 先 warmup, 再开启 profiler 采集, 最后汇总
# ============================================================
int32_results = []
bf16_results = []

# warmup: 不采集数据, 让 NPU 稱定
if WARMUP_ITERS>0:
    print("=" * 60)
    print(f"Warmup: 运行 {WARMUP_ITERS} 轮不采集数据")
    print("=" * 60)
    for i, (M, K, N) in enumerate(TEST_SHAPES, 1):
        test_int32(M, K, N)
        test_bf16(M, K, N)

# 创建 profiler
os.makedirs(PROF_OUTPUT_DIR, exist_ok=True)
profiler = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.NPU,
        torch_npu.profiler.ProfilerActivity.CPU,
    ],
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(PROF_OUTPUT_DIR),
)

# 开启 profiler 采集
profiler.start()

for i, (M, K, N) in enumerate(TEST_SHAPES, 1):
    print("=" * 60)
    print(f"规格{i}: M={M}, K={K}, N={N}")
    print("=" * 60)

    print("[INT32 测试]")
    int32_pass = test_int32(M, K, N)
    int32_results.append((M, K, N, int32_pass))
    profiler.step()

    print("[BF16 测试]")
    bf16_pass = test_bf16(M, K, N)
    bf16_results.append((M, K, N, bf16_pass))
    profiler.step()

# 停止 profiler
profiler.stop()

# ============================================================
# 测试汇总
# ============================================================
print()
print("=" * 60)
print("测试汇总")
print("=" * 60)
print(f"{'规格':>24} | {'INT32':^8} | {'BF16':^8}")
print("-" * 46)
for (M, K, N, p1), (_, _, _, p2) in zip(int32_results, bf16_results):
    tag = f"M={M},K={K},N={N}"
    print(f"{tag:>24} | {'PASS' if p1 else 'FAIL':^8} | {'PASS' if p2 else 'FAIL':^8}")

int32_total = sum(p for _, _, _, p in int32_results)
bf16_total = sum(p for _, _, _, p in bf16_results)
total = len(TEST_SHAPES)
print(f"INT32: {int32_total}/{total} 通过, BF16: {bf16_total}/{total} 通过")

# ============================================================
# Profiler 数据汇总: 每种规格每种 dtype 的 Duration(us)
# ============================================================
print()
print("=" * 70)
print("Profiler 数据汇总 (QmmCustom Duration)")
print("=" * 70)

csv_path = find_kernel_details_csv(f"原始数据路径：{PROF_OUTPUT_DIR}")
print(csv_path)
if csv_path:
    kernel_rows = parse_kernel_details(csv_path)
    if kernel_rows:
        # 按 (M, K, N) 分组, 每组内有 INT32 和 BF16
        shape_map = {}
        for kr in kernel_rows:
            key = (kr["M"], kr["K"], kr["N"])
            shape_map.setdefault(key, {})[kr["dtype"]] = kr["duration_us"]

        print(f"{'规格 (M,K,N)':>24} | {'INT32 Duration(us)':>18} | {'BF16 Duration(us)':>18}")
        print("-" * 70)
        for M, K, N in TEST_SHAPES:
            key = (M, K, N)
            int32_dur = shape_map.get(key, {}).get("INT32", "-")
            bf16_dur = shape_map.get(key, {}).get("BF16", "-")
            int32_str = f"{int32_dur:.3f}" if isinstance(int32_dur, float) else int32_dur
            bf16_str = f"{bf16_dur:.3f}" if isinstance(bf16_dur, float) else bf16_dur
            tag = f"M={M},K={K},N={N}"
            print(f"{tag:>24} | {int32_str:>18} | {bf16_str:>18}")
    else:
        print(f"在 {csv_path} 中未找到 QmmCustom 算子数据")
else:
    print(f"未在 {PROF_OUTPUT_DIR} 中找到 kernel_details.csv 文件")
    print("请检查 profiler 输出目录")

print("=" * 70)
