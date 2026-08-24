# 1. B
# 2. Tensor 计算逻辑
# 3. A
# 4. shape；dtype
# 5. A
# 6. Execute Graph

# 7. 编程题核心代码参考实现：把加法算子改写为 out = (x + y) * 0.5
@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def half_add_kernel(x: pypto.Tensor([], pypto.DT_FP32), y: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(32, 32)
    out[:] = pypto.mul(pypto.add(x, y), 0.5)