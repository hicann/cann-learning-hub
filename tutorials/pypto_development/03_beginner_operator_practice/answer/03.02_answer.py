# 课后实践参考答案：融合乘加 ReLU 后增加平方输出

reset_pypto_notebook_state()

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
    device_local = current_device(device_id)
    x = torch.randn(8, 8, dtype=torch.float16, device=device_local)
    scale = torch.full((8, 8), 1.5, dtype=torch.float16, device=device_local)
    bias = torch.full((8, 8), -0.1, dtype=torch.float16, device=device_local)
    out = torch.empty_like(x)

    fused_square_practice_kernel(x, scale, bias, out)

    ref = torch.maximum(x * scale + bias, torch.zeros_like(x))
    ref = ref * ref
    check_close("fused_square_practice_kernel", out, ref, rtol=1e-3, atol=1e-3)

    print("fused_square_practice_kernel 验证通过")
    print("输出 shape:", tuple(out.shape))
    print("最大误差:", max_abs_diff(out, ref))


main_fused_square_practice()
