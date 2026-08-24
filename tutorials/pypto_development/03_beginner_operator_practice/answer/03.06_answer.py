@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def row_softmax_practice_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    shifted = x - row_max
    exp = pypto.exp(shifted)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    out.move(exp / esum)