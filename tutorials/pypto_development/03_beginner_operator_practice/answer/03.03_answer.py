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
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(8, 8)
    result = pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
    out.move(pypto.maximum(result, 0))