"""
04_quantize_model.py - 静态训练后量化（Static PTQ）
实验4.5 昇腾香橙派部署深度学习网络实验

本脚本对应 lab4.2 的模型量化内容。
对裁剪后的模型进行静态 PTQ 量化，将 FP32 权重压缩为 INT8。

静态 PTQ 三步流程（来自 lab4.2）:
1. prepare: 插入 Observer 观测器
2. calibration: 用校准数据统计激活值范围
3. convert: 替换为量化模块，权重转为 INT8

量化效果: 模型体积约降为 1/4 (FP32 4字节 -> INT8 1字节)

运行方式:
    python 04_quantize_model.py

输出:
    models/student_quantized.pth  - INT8 量化模型
"""

import os
import io
import torch
import torch.nn as nn

from utils import (get_device, get_dataloaders, get_model_size_mb,
                   evaluate_model, setup_quant_engine)
from distillation_helper import load_student_model, load_pruned_model


def model_size_bytes(model):
    """统计模型 state_dict 的字节大小"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


class _QuantWrapper(nn.Module):
    """为不含 QuantStub 的模型补上输入/输出量化节点 (eager PTQ 必需)

    PyTorch eager 模式静态量化要求模型输入处有 QuantStub、输出处有
    DeQuantStub，否则 convert 后首个量化算子会收到 float32 张量而报
    "quantized::conv2d ... CPU backend" 错误。未 prepare 时这两个节点
    为恒等映射，不影响前向结果；prepare/convert 后会被替换为真正的
    Quantize/DeQuantize 算子。
    """

    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def quantize_static(model, calibration_loader, engine='qnnpack'):
    """静态 PTQ 量化

    步骤:
    1. 设置 qconfig (量化配置)
    2. prepare: 插入 Observer
    3. 校准: 前向传播让 Observer 统计激活范围
    4. convert: 转换为 INT8 量化模型

    Args:
        model: FP32 模型
        calibration_loader: 校准数据 DataLoader
        engine: 量化后端 ('qnnpack' for ARM, 'fbgemm' for x86)
    Returns:
        量化后的 INT8 模型
    """
    model = model.cpu().eval()
    model = _QuantWrapper(model)

    torch.backends.quantized.engine = engine
    model.qconfig = torch.quantization.get_default_qconfig(engine)

    # 步骤1: prepare - 插入 Observer
    torch.quantization.prepare(model, inplace=True)
    print("prepare 完成: Observer 已插入")

    # 步骤2: 校准 - 用数据前向传播统计激活范围
    print("开始校准...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            model(images)
            if i >= 5:
                break
    print(f"校准完成 ({min(i+1, 6)} 批次)")

    # 步骤3: convert - 转换为 INT8 模型
    torch.quantization.convert(model, inplace=True)
    print("convert 完成: 模型已量化为 INT8")

    return model


def main():
    print("=" * 60)
    print("步骤4: 静态训练后量化 (Static PTQ)")
    print("=" * 60)

    # 量化在 CPU 上进行
    device = torch.device('cpu')
    print(f"量化设备: {device} (PyTorch eager量化仅支持CPU)")

    # 选择量化后端
    engine = setup_quant_engine()
    print(f"量化后端: {engine}")

    train_loader, test_loader, _, _ = get_dataloaders(
        image_dir='./images', augment_times=50, batch_size=16)

    # 加载裁剪后的模型
    model = load_pruned_model(device)
    model = model.cpu().eval()

    # 量化前
    size_before = get_model_size_mb(model)
    acc_before = evaluate_model(model, test_loader, device)
    print(f"\n量化前 (FP32): 大小={size_before:.2f} MB, 准确率={acc_before*100:.1f}%")

    # 执行静态量化
    print("\n执行静态 PTQ 量化...")
    model_quantized = quantize_static(model, train_loader, engine=engine)

    # 量化后
    size_after = get_model_size_mb(model_quantized)
    acc_after = evaluate_model(model_quantized, test_loader, device)
    print(f"\n量化后 (INT8): 大小={size_after:.2f} MB, 准确率={acc_after*100:.1f}%")

    # 保存
    os.makedirs('./models', exist_ok=True)
    torch.save(model_quantized.state_dict(), './models/student_quantized.pth')
    print(f"量化模型已保存到 ./models/student_quantized.pth")

    # 总结
    print("\n" + "=" * 50)
    print("量化结果总结:")
    print(f"  量化前 (FP32): {size_before:.2f} MB, 准确率={acc_before*100:.1f}%")
    print(f"  量化后 (INT8): {size_after:.2f} MB, 准确率={acc_after*100:.1f}%")
    print(f"  体积压缩比:    {size_before/size_after:.2f}x")
    print(f"  精度变化:       {(acc_after-acc_before)*100:+.1f}%")
    print("=" * 50)

    return model_quantized


if __name__ == '__main__':
    main()
