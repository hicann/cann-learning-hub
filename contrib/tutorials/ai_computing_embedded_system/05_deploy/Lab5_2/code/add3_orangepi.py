"""
阶段三：三向量加法 —— 在昇腾香橙派 310B 上运行
算子：z = x + y + w
输入 shape: (8, 2048)，共 16384 个元素
输入 x 全部填充 1.2，y 全部填充 2.3，w 全部填充 3.4
预期输出 z 全部为 6.9 (1.2 + 2.3 + 3.4)
运行方式：python3 add3_orangepi.py
"""

import torch
import torch_npu


def verify_result(output, golden):
    """精度验证：逐元素比对输出与 golden 值"""
    print(f"Output: {' '.join(f'{v:.1f}' for v in output[:20].tolist())}...")
    print(f"Golden: {' '.join(f'{v:.1f}' for v in golden[:20].tolist())}...")
    if torch.allclose(output, golden, rtol=1e-6, atol=1e-6):
        print("[Success] 精度验证通过！")
        return 0
    else:
        print("[Failed] 精度验证失败！")
        return 1


def main():
    # ===== 数据参数定义 =====
    # 与阶段一的区别：输入数量 2 -> 3，计算公式 z = x + y -> z = x + y + w
    use_core_num = 8
    block_length = 2048
    total_length = use_core_num * block_length  # 16384
    value_x = 1.2
    value_y = 2.3
    value_w = 3.4

    # ===== NPU 设备初始化 =====
    device = torch.device('npu:0')
    print(f"计算设备: {device}")

    # ===== 输入数据生成（Host 侧）=====
    # 增加第三个输入向量 w
    x = torch.full((total_length,), value_x, dtype=torch.float32)
    y = torch.full((total_length,), value_y, dtype=torch.float32)
    w = torch.full((total_length,), value_w, dtype=torch.float32)

    # ===== 数据搬移到 NPU（Host -> Device）=====
    x_npu = x.to(device)
    y_npu = y.to(device)
    w_npu = w.to(device)

    # ===== NPU 上执行三向量加法 =====
    # 对应 Ascend C Compute 函数中的两次 Add：
    #   第 1 次：Add(zLocal, xLocal, yLocal, tileLength)  -> z = x + y = 3.5
    #   第 2 次：Add(zLocal, zLocal, wLocal, tileLength)  -> z = z + w = 6.9
    # PyTorch 中可直接写 x + y + w，底层分两次加法指令执行
    z_npu = x_npu + y_npu + w_npu

    # ===== 结果搬回 Host（Device -> Host）=====
    z = z_npu.cpu()

    # ===== Golden 计算与精度验证 =====
    golden = torch.full((total_length,), value_x + value_y + value_w, dtype=torch.float32)
    return verify_result(z, golden)


if __name__ == '__main__':
    main()
