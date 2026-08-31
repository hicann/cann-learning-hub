"""
03_prune_model.py - L1 非结构化裁剪
实验4.5 昇腾香橙派部署深度学习网络实验

本脚本对应 lab4.3 的网络裁剪内容。
对训练好的学生模型进行 L1 非结构化裁剪，将绝对值最小的权重置零。

裁剪核心思想（来自 lab4.3）:
- 权重绝对值越小，对输出贡献越小，越可以裁掉
- L1 非结构化裁剪: 按绝对值排序，将最小的 amount 比例的权重置零
- 裁剪后需要微调恢复精度

运行方式:
    python 03_prune_model.py

输出:
    models/student_pruned.pth  - 裁剪+微调后的模型
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from utils import (get_device, get_dataloaders, count_parameters,
                   get_model_size_mb, evaluate_model, train_model,
                   check_sparsity)
from distillation_helper import load_student_model


def apply_pruning(model, amount=0.3):
    """对模型中所有 Conv2d 和 Linear 层做 L1 非结构化裁剪

    API: torch.nn.utils.prune.l1_unstructured(module, name='weight', amount)
    - 按权重绝对值排序，将最小的 amount 比例置零
    - prune.remove 将掩码永久化写入权重

    Args:
        model: 待裁剪模型
        amount: 裁剪比例 (0~1)，如 0.3 表示裁掉 30% 最小权重
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    print(f"裁剪完成 (amount={amount})")


def main():
    print("=" * 60)
    print("步骤3: L1 非结构化裁剪")
    print("=" * 60)

    device, has_npu = get_device()
    print(f"计算设备: {device}")

    train_loader, test_loader, _, _ = get_dataloaders(
        image_dir='./images', augment_times=50, batch_size=16)

    # 加载训练好的学生模型
    model = load_student_model(device)
    model_before = copy.deepcopy(model)

    # 裁剪前评估
    acc_before = evaluate_model(model_before, test_loader, device)
    size_before = get_model_size_mb(model_before)
    print(f"\n裁剪前: 准确率={acc_before*100:.1f}%, 大小={size_before:.2f} MB")

    # 执行裁剪
    print("\n执行 L1 非结构化裁剪 (amount=0.3)...")
    apply_pruning(model, amount=0.3)

    # 检查稀疏率
    sparsity = check_sparsity(model)

    # 裁剪后评估
    acc_after = evaluate_model(model, test_loader, device)
    size_after = get_model_size_mb(model)
    print(f"裁剪后: 准确率={acc_after*100:.1f}%, 大小={size_after:.2f} MB")
    print(f"  (非结构化裁剪体积不变是正常的，需稀疏存储格式才减小)")

    # 微调恢复精度
    print("\n微调恢复精度...")
    train_model(model, train_loader, test_loader, device,
                num_epochs=3, learning_rate=0.0005,
                model_name='Pruned-Finetune')
    acc_finetune = evaluate_model(model, test_loader, device)
    print(f"微调后: 准确率={acc_finetune*100:.1f}%")

    # 保存
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/student_pruned.pth')
    print(f"\n裁剪模型已保存到 ./models/student_pruned.pth")

    # 总结
    print("\n" + "=" * 50)
    print("裁剪结果总结:")
    print(f"  裁剪前准确率:  {acc_before*100:.1f}%")
    print(f"  裁剪后准确率:  {acc_after*100:.1f}%")
    print(f"  微调后准确率:  {acc_finetune*100:.1f}%")
    print(f"  稀疏率:        {sparsity:.2f}%")
    print(f"  模型大小:      {size_before:.2f} MB (非结构化裁剪不变)")
    print("=" * 50)

    return model


if __name__ == '__main__':
    main()
