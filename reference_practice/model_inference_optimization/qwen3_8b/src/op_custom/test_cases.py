import torch
import torch_npu
import custom_ops

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


# ============================================================
# 主循环: 对每组 (M, K, N) 执行 INT32 和 BF16 测试
# ============================================================
int32_results = []
bf16_results = []

for i, (M, K, N) in enumerate(TEST_SHAPES, 1):
    print("=" * 60)
    print(f"规格{i}: M={M}, K={K}, N={N}")
    print("=" * 60)

    print("[INT32 测试]")
    int32_pass = test_int32(M, K, N)
    int32_results.append((M, K, N, int32_pass))

    print("[BF16 测试]")
    bf16_pass = test_bf16(M, K, N)
    bf16_results.append((M, K, N, bf16_pass))

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
print("=" * 60)
