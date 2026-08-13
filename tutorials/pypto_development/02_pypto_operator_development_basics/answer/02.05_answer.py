# 1. B
# 2. Tensor 计算逻辑
# 3. A
# 4. shape；dtype
# 5. A
# 6. Execute Graph

# 7. 编程题核心代码参考实现：把加法算子改写为 out = (x + y) * 0.5
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def half_add_kernel(x: pypto.Tensor[...], y: pypto.Tensor[...], out: pypto.Tensor[...]):
    pypto.set_vec_tile_shapes(32, 32)
    out[:] = pypto.mul(pypto.add(x, y), 0.5)