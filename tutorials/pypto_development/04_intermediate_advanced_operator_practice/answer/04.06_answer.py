# 章节实践参考答案：动态 Residual + LayerNorm + GELU 近似

# 章节自测答案：
# 1. A
# 2. 真实有效的数据范围
# 3. A
# 4. LayerNorm；GELU
# 5. A

# 6. 编程题参考实现
@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def dynamic_residual_norm_gelu_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, HIDDEN_SIZE], pypto.DT_FP32),
    residual_input: pypto.Tensor([pypto.DYNAMIC, HIDDEN_SIZE], pypto.DT_FP32),
    gamma: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    beta: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, HIDDEN_SIZE], pypto.DT_FP32),
):
    tile_b = 8
    batch = x.shape[0]
    pypto.set_vec_tile_shapes(tile_b, HIDDEN_SIZE)
    b_loop = (batch + tile_b - 1) // tile_b
    hidden_size_inv = 1.0 / HIDDEN_SIZE
    for b_idx in pypto.loop(b_loop):
        b_offset = b_idx * tile_b
        b_end = min(b_offset + tile_b, batch)
        valid_shape = [b_end - b_offset, HIDDEN_SIZE]
        x_view = pypto.view(x, [tile_b, HIDDEN_SIZE], [b_offset, 0], valid_shape=valid_shape)
        residual_view = pypto.view(
            residual_input,
            [tile_b, HIDDEN_SIZE],
            [b_offset, 0],
            valid_shape=valid_shape,
        )
        residual = x_view + residual_view
        mean = pypto.mul(pypto.sum(residual, dim=-1, keepdim=True), hidden_size_inv)
        centered = residual - mean
        var = pypto.mul(pypto.sum(centered * centered, dim=-1, keepdim=True), hidden_size_inv)
        normalized = centered / pypto.sqrt(pypto.add(var, EPS))
        scaled = normalized * gamma + beta
        activated = scaled * pypto.sigmoid(pypto.mul(scaled, GELU_COEFF))
        pypto.assemble(activated, [b_offset, 0], out)