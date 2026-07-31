# 课后实践参考答案：矩阵乘法 + Bias + ReLU
#
# 自测题答案：
# 1. B
# 2. 逐元素
# 3. A

@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def matmul_bias_relu_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)):
    extend_params = {"bias_tensor": bias}
    # Cube Tile Shape（矩阵乘法使用）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    # Vector Tile Shape（Maximum/ReLU 使用）—— 修正接口名和参数
    pypto.set_vec_tile_shapes(8, 8)
    # 第一步：matmul + bias
    result = pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
    # 第二步：ReLU（maximum(result, 0)）
    out.move(pypto.maximum(result, 0))


def test():
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device=device)
    bias = torch.tensor([[1, 2]], dtype=torch.float32, device=device)
    out = torch.empty((2, 2), dtype=torch.float32, device=device)
    matmul_bias_relu_kernel(a, b, bias, out)
    ref = torch.maximum(torch.matmul(a, b) + bias, torch.tensor(0.0))
    max_diff = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("matmul_bias_relu_kernel 验证通过")
    print("device:", device, "run_mode:", RUN_MODE)
    print("a shape:", tuple(a.shape), "b shape:", tuple(b.shape), "bias shape:", tuple(bias.shape))
    print("输出:", out.cpu())
    print("参考:", ref.cpu())
    print("最大误差:", max_diff)

test()
