"""
distillation_helper.py - 模型加载辅助模块
实验4.5 昇腾香橙派部署深度学习网络实验

提供学生模型的定义和加载函数，供其他脚本共用。
"""

import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class StudentCNN(nn.Module):
    """小型学生网络，用于猫狗分类

    结构: Conv(3→16) → BN → ReLU → Conv(16→32) → BN → ReLU → Pool
          → Conv(32→64) → BN → ReLU → Pool → GAP → FC
    参数量约 0.2M，远小于 MobileNetV2 的 3.5M
    适合在昇腾香橙派 (Ascend 310B) 上部署推理
    """

    def __init__(self, num_classes=2):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_student_model(device, model_path='./models/student_distilled.pth'):
    """加载蒸馏训练好的学生模型"""
    model = StudentCNN(num_classes=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"已加载学生模型: {model_path}")
    else:
        print(f"警告: {model_path} 不存在，使用随机初始化权重")
    model = model.to(device)
    return model


def load_pruned_model(device, model_path='./models/student_pruned.pth'):
    """加载裁剪后的学生模型"""
    model = StudentCNN(num_classes=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"已加载裁剪模型: {model_path}")
    else:
        print(f"警告: {model_path} 不存在，尝试加载蒸馏模型")
        return load_student_model(device)
    model = model.to(device)
    return model


def load_mobilenetv2(device, model_path='./models/mobilenet_v2_catdog.pth'):
    """加载训练好的 MobileNetV2 模型"""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"已加载 MobileNetV2: {model_path}")
    model = model.to(device)
    return model
