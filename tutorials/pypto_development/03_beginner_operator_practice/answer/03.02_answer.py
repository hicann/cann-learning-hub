# 课后实践核心代码实现：融合乘加 ReLU 后增加平方输出
@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def fused_square_practice_kernel(
    x: pypto.Tensor([], pypto.DT_FP16),
    scale: pypto.Tensor([], pypto.DT_FP16),
    bias: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(8, 8)
    y = pypto.add(pypto.mul(x, scale), bias)
    relu = pypto.maximum(y, 0.0)
    squared = pypto.mul(relu, relu)
    out.move(squared)


def main_fused_square_practice(device_id: int = None):
    device_local = get_device()
    x = torch.randn(8, 8, dtype=torch.float16, device=device_local)
    scale = torch.full((8, 8), 1.5, dtype=torch.float16, device=device_local)
    bias = torch.full((8, 8), -0.1, dtype=torch.float16, device=device_local)
    out = torch.empty_like(x)

    fused_square_practice_kernel(x, scale, bias, out)

    ref = torch.maximum(x * scale + bias, torch.zeros_like(x))
    ref = ref * ref
    
    torch.testing.assert_close(out, ref, rtol=3e-3, atol=3e-3)
    max_diff = (out - ref).abs().max().item()
    print(f"最大误差: {max_diff:.6f}")
