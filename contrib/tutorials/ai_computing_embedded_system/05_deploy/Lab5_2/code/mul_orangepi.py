"""
阶段四：双向量乘法 —— 在昇腾香橙派 310B 上运行
算子：z = x * y
输入 shape: (8, 2048)，共 16384 个元素
输入 x 全部填充 1.2，输入 y 全部填充 2.3
预期输出 z 全部为 2.76 (1.2 * 2.3)
运行方式：python3 mul_orangepi.py
"""

import torch
import torch_npu


def verify_result(output, golden):
    """精度验证：逐元素比对输出与 golden 值"""
    print(f"Output: {' '.join(f'{v:.2f}' for v in output[:20].tolist())}...")
    print(f"Golden: {' '.join(f'{v:.2f}' for v in golden[:20].tolist())}...")
    if torch.allclose(output, golden, rtol=1e-5, atol=1e-5):
        print("[Success] 精度验证通过！")
        return 0
    else:
        print("[Failed] 精度验证失败！")
        return 1


def main():
    # ===== 数据参数定义 =====
    # 与阶段一的区别：计算指令 Add -> Mul，golden 计算 + -> *
    use_core_num = 8
    block_length = 2048
    total_length = use_core_num * block_length  # 16384
    value_x = 1.2
    value_y = 2.3

    # ===== NPU 设备初始化 =====
    device = torch.device('npu:0')
    print(f"计算设备: {device}")

    # ===== 输入数据生成（Host 侧）=====
    x = torch.full((total_length,), value_x, dtype=torch.float32)
    y = torch.full((total_length,), value_y, dtype=torch.float32)

    # ===== 数据搬移到 NPU（Host -> Device）=====
    x_npu = x.to(device)
    y_npu = y.to(device)

    # ===== NPU 上执行向量乘法 =====
    # 对应 Ascend C Compute 函数中的 Mul(zLocal, xLocal, yLocal, tileLength)
    # 关键变化：Add 指令替换为 Mul 指令
    z_npu = x_npu * y_npu

    # ===== 结果搬回 Host（Device -> Host）=====
    z = z_npu.cpu()

    # ===== Golden 计算与精度验证 =====
    # 关键变化：golden 从 value_x + value_y 改为 value_x * value_y
    golden = torch.full((total_length,), value_x * value_y, dtype=torch.float32)
    return verify_result(z, golden)


if __name__ == '__main__':
    main()
