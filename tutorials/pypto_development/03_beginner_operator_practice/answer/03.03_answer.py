# 课后实践参考答案：矩阵乘法 + Bias + ReLU
#
# 自测题答案：
# 1. B
# 2. 逐元素
# 3. A

reset_pypto_notebook_state()

@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def matmul_bias_relu_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)):
    extend_params = {"bias_tensor": bias}
    pypto.set_cube_tile_shapes((2, 8), (8, 8), (2, 8))
    with_bias = pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
    pypto.set_vec_tile_shapes(2, 8)
    relu = pypto.maximum(with_bias, 0.0)
    out.move(relu)


def main_matmul_bias_relu(device_id: int = None):
    device_local = current_device(device_id)
    a = torch.randn(2, 8, dtype=torch.float32, device=device_local)
    b = torch.randn(8, 8, dtype=torch.float32, device=device_local)
    bias = torch.randn(1, 8, dtype=torch.float32, device=device_local)
    out = torch.empty((2, 8), dtype=torch.float32, device=device_local)

    matmul_bias_relu_kernel(a, b, bias, out)

    ref = torch.maximum(torch.matmul(a, b) + bias, torch.zeros_like(out))
    check_close("matmul_bias_relu_kernel", out, ref, rtol=1e-3, atol=1e-3)

    print("matmul_bias_relu_kernel 验证通过")
    print("输出 shape:", tuple(out.shape))
    print("最大误差:", max_abs_diff(out, ref))


main_matmul_bias_relu()
