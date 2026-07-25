# 课后实践参考答案：按行中心化

# 自测题答案：
# 1. B
# 2. `keepdim=True` 保留被规约的维度，方便后续广播。
# 3. A
# 4. 提升数值稳定性，降低指数计算溢出风险。

reset_pypto_notebook_state()

@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def row_center_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    row_sum = pypto.sum(x, dim=-1, keepdim=True)
    mean = row_sum / 8.0
    out.move(x - mean)


def main_row_center(device_id: int = None):
    device_local = current_device(device_id)
    x = torch.randn(8, 8, dtype=torch.float32, device=device_local)
    out = torch.empty_like(x)

    row_center_kernel(x, out)

    ref = x - torch.mean(x, dim=-1, keepdim=True)
    check_close("row_center_kernel", out, ref, rtol=1e-3, atol=1e-3)

    print("row_center_kernel 验证通过")
    print("输出 shape:", tuple(out.shape))
    print("输出每行均值:", out.mean(dim=-1).detach().cpu())
    print("最大误差:", max_abs_diff(out, ref))


main_row_center()
