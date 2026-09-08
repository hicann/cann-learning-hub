"""
阶段一：双向量加法（8核）—— 在昇腾香橙派 310B 上运行
算子：z = x + y
输入 shape: (8, 2048)，共 16384 个元素
输入 x 全部填充 1.2，输入 y 全部填充 2.3
预期输出 z 全部为 3.5 (1.2 + 2.3)
运行方式：python3 add_8core_orangepi.py
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
    # 对应 Ascend C 算子中的常量定义
    # totalLength = 8 * 2048 = 16384，平均分配到 8 个核，每核处理 2048 个元素
    use_core_num = 8
    block_length = 2048
    total_length = use_core_num * block_length  # 16384
    value_x = 1.2
    value_y = 2.3

    # ===== NPU 设备初始化 =====
    # 香橙派 310B 使用 npu 设备，对应 Ascend C 中的 aclrtSetDevice
    device = torch.device('npu:0')
    print(f"计算设备: {device}")

    # ===== 输入数据生成（Host 侧）=====
    # 对应 Ascend C main 函数中的 std::vector<float> x(totalLength, valueX)
    x = torch.full((total_length,), value_x, dtype=torch.float32)
    y = torch.full((total_length,), value_y, dtype=torch.float32)

    # ===== 数据搬移到 NPU（Host -> Device）=====
    # 对应 Ascend C 中的 aclrtMemcpy(xDevice, ..., xHost, ..., ACL_MEMCPY_HOST_TO_DEVICE)
    x_npu = x.to(device)
    y_npu = y.to(device)

    # ===== NPU 上执行向量加法 =====
    # 对应 Ascend C 核函数中的 Add(zLocal, xLocal, yLocal, tileLength)
    # 底层由 8 个 AI Core 并行处理，每个核处理 2048 个元素
    z_npu = x_npu + y_npu

    # ===== 结果搬回 Host（Device -> Host）=====
    # 对应 Ascend C 中的 aclrtMemcpy(zHost, ..., zDevice, ..., ACL_MEMCPY_DEVICE_TO_HOST)
    z = z_npu.cpu()

    # ===== Golden 计算与精度验证 =====
    golden = torch.full((total_length,), value_x + value_y, dtype=torch.float32)
    return verify_result(z, golden)


if __name__ == '__main__':
    main()
