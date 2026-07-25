# 03.06 章节实践参考答案：实现稳定版行 Softmax

# 章节自测答案：
# 1. 逐元素、规约等向量
# 2. 矩阵乘法等 Cube
# 3. B
# 4. A
# 5. A
# 6. 把计算表达式的最终结果写回调用者传入的输出 Tensor

reset_pypto_notebook_state()


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


def main_row_softmax_practice():
    if device == "cpu":
        print("当前环境未执行 NPU 验证；NPU 环境中可执行本模块。")
        return

    x = torch.randn((8, 8), dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    row_softmax_practice_kernel(x, out)

    ref = torch.softmax(x, dim=-1)
    max_diff = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    print("row_softmax_practice_kernel 验证通过")
    print("输入 shape:", tuple(x.shape), "输出 shape:", tuple(out.shape))
    print("输出每行求和:", out.sum(dim=-1).detach().cpu())
    print("最大误差:", max_diff)


main_row_softmax_practice()
