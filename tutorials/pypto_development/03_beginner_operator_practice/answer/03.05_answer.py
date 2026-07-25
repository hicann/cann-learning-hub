# 课后实践参考答案：切分 Q/K/V 并转置 K

# 自测题答案：
# 1. B
# 2. 元素总数
# 3. A

reset_pypto_notebook_state()

@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def split_qkv_kernel(
    x: pypto.Tensor([], pypto.DT_FP16),
    q_out: pypto.Tensor([], pypto.DT_FP16),
    k_out: pypto.Tensor([], pypto.DT_FP16),
    v_out: pypto.Tensor([], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(2, 8, 64)
    q = x[:, :, :64]
    k = x[:, :, 64:128]
    v = x[:, :, 128:192]
    q_out.move(q)
    k_out.move(pypto.transpose(k, 1, 2))
    v_out.move(v)


def main_split_qkv(device_id: int = None):
    device_local = current_device(device_id)
    x = torch.randn(2, 8, 192, dtype=torch.float16, device=device_local)
    q_out = torch.empty(2, 8, 64, dtype=torch.float16, device=device_local)
    k_out = torch.empty(2, 64, 8, dtype=torch.float16, device=device_local)
    v_out = torch.empty(2, 8, 64, dtype=torch.float16, device=device_local)

    split_qkv_kernel(x, q_out, k_out, v_out)

    q_ref = x[:, :, :64]
    k_ref = x[:, :, 64:128].transpose(1, 2)
    v_ref = x[:, :, 128:192]
    check_close("split_qkv q", q_out, q_ref, rtol=1e-3, atol=1e-3)
    check_close("split_qkv k", k_out, k_ref, rtol=1e-3, atol=1e-3)
    check_close("split_qkv v", v_out, v_ref, rtol=1e-3, atol=1e-3)

    print("split_qkv_kernel 验证通过")
    print("Q/K/V shape:", tuple(q_out.shape), tuple(k_out.shape), tuple(v_out.shape))


main_split_qkv()
