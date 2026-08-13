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

