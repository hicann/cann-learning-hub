"""
阶段二：双向量加法（32核）—— 在昇腾香橙派 310B 上运行
算子：z = x + y
输入 shape: (32, 2048)，共 65536 个元素
输入 x 全部填充 2.2，输入 y 全部填充 2.3
预期输出 z 全部为 4.5 (2.2 + 2.3)
运行方式：python3 add_32core_orangepi.py
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
    # 与阶段一的区别：核数 8 -> 32，数据总量 16384 -> 65536
    # 每核处理数据量保持不变：2048 个元素
    use_core_num = 32
    block_length = 2048
    total_length = use_core_num * block_length  # 65536
    value_x = 2.2  # 与阶段一的 1.2 区分
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

    # ===== NPU 上执行向量加法 =====
    # 32 个 AI Core 并行处理，每个核处理 2048 个元素
    z_npu = x_npu + y_npu

    # ===== 结果搬回 Host（Device -> Host）=====
    z = z_npu.cpu()

    # ===== Golden 计算与精度验证 =====
    golden = torch.full((total_length,), value_x + value_y, dtype=torch.float32)
    return verify_result(z, golden)


if __name__ == '__main__':
    main()
