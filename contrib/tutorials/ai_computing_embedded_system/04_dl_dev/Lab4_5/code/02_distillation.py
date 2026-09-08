"""
02_distillation.py - 知识蒸馏：用大模型训练小模型
实验4.5 昇腾香橙派部署深度学习网络实验

本脚本对应 lab4.4 的知识蒸馏内容。
教师网络: MobileNetV2（参数多，精度高）
学生网络: 小型自定义CNN（参数少，推理快，适合香橙派）

蒸馏核心思想:
- 教师输出包含"暗知识"（类间关系信息）
- 温度软化(T>1)使分布平滑，暴露暗知识
- 蒸馏损失 = α·硬标签损失 + (1-α)·T²·软标签损失

运行方式:
    python 02_distillation.py

输出:
    models/teacher_mobilenetv2.pth  - 教师模型
    models/student_distilled.pth    - 蒸馏后的学生模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utils import (get_device, get_dataloaders, count_parameters,
                   get_model_size_mb, evaluate_model, train_model)


# ============================================================
# 学生网络定义（小型CNN，适合香橙派部署）
# ============================================================
class StudentCNN(nn.Module):
    """小型学生网络，用于猫狗分类

    结构: Conv(3→16) → Conv(16→32) → Pool → Conv(32→64) → Pool → FC
    参数量约 0.2M，远小于 MobileNetV2 的 3.5M
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
            nn.AvgPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),  # 112 -> 56
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


# ============================================================
# 蒸馏损失函数
# ============================================================
def distillation_loss(student_logits, teacher_logits, labels,
                      T=4.0, alpha=0.5):
    """知识蒸馏损失函数（来自 lab4.4）

    L = α · L_hard + (1 - α) · T² · L_soft

    Args:
        student_logits: 学生网络输出 logits
        teacher_logits: 教师网络输出 logits（已 detach）
        labels: 真实标签
        T: 蒸馏温度（越大分布越平滑，常用 2~10）
        alpha: 硬损失权重（0~1，常用 0.5）
    Returns:
        total_loss, hard_loss, soft_loss
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher,
                         reduction='batchmean') * (T * T)

    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
    return total_loss, hard_loss, soft_loss


# ============================================================
# 蒸馏训练函数
# ============================================================
def distill_train(teacher, student, train_loader, test_loader, device,
                  num_epochs=5, lr=0.001, T=4.0, alpha=0.5):
    """蒸馏训练学生网络

    关键步骤:
    1. 教师设为 eval()，只做推理
    2. 学生设为 train()，正常前向+反向
    3. 每个 batch: 学生前向 + 教师前向(no_grad) + 蒸馏损失 + 反传
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    teacher.eval()
    print(f"\n开始蒸馏训练 (T={T}, alpha={alpha})...")

    for epoch in range(num_epochs):
        student.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            loss, hard_l, soft_l = distillation_loss(
                student_logits, teacher_logits, labels, T=T, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            running_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        test_acc = evaluate_model(student, test_loader, device)

        print(f'  [Distill] Epoch {epoch+1}/{num_epochs}: '
              f'Loss={train_loss:.4f} TrainAcc={train_acc:.4f} '
              f'TestAcc={test_acc:.4f}')

    return student


def main():
    print("=" * 60)
    print("步骤2: 知识蒸馏 - 用MobileNetV2训练小型StudentCNN")
    print("=" * 60)

    device, has_npu = get_device()
    print(f"计算设备: {device}")

    train_loader, test_loader, _, _ = get_dataloaders(
        image_dir='./images', augment_times=50, batch_size=16)

    # --- 教师网络: MobileNetV2 ---
    teacher = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, 2)
    teacher = teacher.to(device)

    print("\n训练教师网络 (MobileNetV2)...")
    train_model(teacher, train_loader, test_loader, device,
                num_epochs=5, learning_rate=0.001, model_name='Teacher')

    teacher_acc = evaluate_model(teacher, test_loader, device)
    teacher_params = count_parameters(teacher)
    print(f"教师准确率: {teacher_acc*100:.1f}%, 参数量: {teacher_params:,}")

    # --- 学生网络: StudentCNN ---
    student = StudentCNN(num_classes=2).to(device)
    student_params = count_parameters(student)
    print(f"\n学生网络参数量: {student_params:,}")
    print(f"参数压缩比: 教师/学生 = {teacher_params/student_params:.1f}x")

    # --- 蒸馏训练 ---
    student = distill_train(teacher, student, train_loader, test_loader,
                            device, num_epochs=5, lr=0.001, T=4.0, alpha=0.5)

    distill_acc = evaluate_model(student, test_loader, device)

    # --- 基线对比: 无蒸馏训练 ---
    print("\n训练基线学生网络 (无蒸馏)...")
    student_baseline = StudentCNN(num_classes=2).to(device)
    train_model(student_baseline, train_loader, test_loader, device,
                num_epochs=5, learning_rate=0.001, model_name='Student-Baseline')
    baseline_acc = evaluate_model(student_baseline, test_loader, device)

    # --- 对比结果 ---
    print("\n" + "=" * 50)
    print("蒸馏对比结果:")
    print(f"  教师准确率:        {teacher_acc*100:.1f}%")
    print(f"  学生(蒸馏)准确率: {distill_acc*100:.1f}%")
    print(f"  学生(基线)准确率: {baseline_acc*100:.1f}%")
    print(f"  蒸馏收益:          {(distill_acc-baseline_acc)*100:+.1f}%")
    print("=" * 50)

    # --- 保存模型 ---
    os.makedirs('./models', exist_ok=True)
    torch.save(teacher.state_dict(), './models/teacher_mobilenetv2.pth')
    torch.save(student.state_dict(), './models/student_distilled.pth')
    print("模型已保存到 ./models/")

    return teacher, student


if __name__ == '__main__':
    main()
