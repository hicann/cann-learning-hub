# 课后实践参考答案：把加法改为乘加形式

# Kernel 侧计算表达式改为：
#
#     out[:] = (x + y) * 2
#
# Host 侧 PyTorch 参考结果同步改为：
#
#     golden = torch.add(input_data0, input_data1) * 2
#
# 验证逻辑保持不变：
#
#     assert_allclose(output_np, golden_np, rtol=3e-3, atol=3e-3)
