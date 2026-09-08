"""
01_train_mobilenetv2.py - 训练 MobileNetV2 猫狗分类模型
实验4.5 昇腾香橙派部署深度学习网络实验

本脚本对应 lab4.1 的轻量化网络训练内容。
使用 MobileNetV2 预训练权重 + 迁移学习，在猫狗分类数据集上微调。
MobileNetV2 参数量约 3.5M，模型体积约 14MB，适合端侧部署。

运行方式:
    python 01_train_mobilenetv2.py

输出:
    models/mobilenet_v2_catdog.pth  - 训练好的模型权重
"""

import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utils import (get_device, get_dataloaders, count_parameters,
                   get_model_size_mb, train_model, evaluate_model,
                   measure_inference_time)


def build_mobilenetv2(num_classes=2):
    """构建 MobileNetV2 模型并替换分类层

    MobileNetV2 核心思想（来自 lab4.1）:
    - 深度可分离卷积（Depthwise Separable Convolution）
    - 倒残差结构（Inverted Residual）：瘦→胖→瘦
    - 参数量约 3.5M，约为 ResNet18 的 1/3

    Args:
        num_classes: 分类数（猫狗分类=2）
    Returns:
        model: MobileNetV2 模型
    """
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main():
    print("=" * 60)
    print("步骤1: 训练 MobileNetV2 猫狗分类模型")
    print("=" * 60)

    # 设备检测
    device, has_npu = get_device()
    print(f"计算设备: {device}")

    # 数据加载
    train_loader, test_loader, train_ds, test_ds = get_dataloaders(
        image_dir='./images', augment_times=50, batch_size=16)
    print(f"训练样本数: {len(train_ds)}, 测试样本数: {len(test_ds)}")

    # 构建模型
    model = build_mobilenetv2(num_classes=2)
    model = model.to(device)

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"MobileNetV2 参数量: {params:,}")
    print(f"MobileNetV2 模型大小: {size_mb:.2f} MB")

    # 训练
    print("\n开始训练...")
    history = train_model(model, train_loader, test_loader, device,
                          num_epochs=5, learning_rate=0.001,
                          model_name='MobileNetV2')

    # 评估
    acc = evaluate_model(model, test_loader, device)
    inf_time = measure_inference_time(model, test_loader, device)
    print(f"\n最终测试准确率: {acc*100:.1f}%")
    print(f"平均推理时间: {inf_time:.2f} ms")

    # 保存模型
    os.makedirs('./models', exist_ok=True)
    save_path = './models/mobilenet_v2_catdog.pth'
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

    return model, history


if __name__ == '__main__':
    main()
