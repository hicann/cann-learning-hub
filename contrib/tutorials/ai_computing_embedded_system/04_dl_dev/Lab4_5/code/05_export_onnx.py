"""
05_export_onnx.py - 导出 ONNX 模型
实验4.5 昇腾香橙派部署深度学习网络实验

将训练好的 PyTorch 模型导出为 ONNX 格式，
后续使用 ATC 工具将 ONNX 转换为昇腾 OM 离线模型。

ONNX (Open Neural Network Exchange) 是开放的模型交换格式，
是 PyTorch 到昇腾 NPU 部署的桥梁。

运行方式:
    python 05_export_onnx.py

输出:
    models/student_model.onnx  - ONNX 模型文件
"""

import os
import torch
import torch.nn as nn

from utils import get_device, get_dataloaders, count_parameters
from distillation_helper import load_student_model


def export_onnx(model, output_path, input_size=(1, 3, 224, 224)):
    """将 PyTorch 模型导出为 ONNX 格式

    Args:
        model: PyTorch 模型 (eval模式)
        output_path: 输出 ONNX 文件路径
        input_size: 输入张量形状
    """
    model = model.cpu().eval()

    dummy_input = torch.randn(*input_size)

    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "导出 ONNX 需要 onnx 包，请先安装: pip install onnx"
        ) from e

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX 模型已导出到: {output_path}")
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX 文件大小: {file_size:.2f} MB")


def main():
    print("=" * 60)
    print("步骤5: 导出 ONNX 模型")
    print("=" * 60)

    device = torch.device('cpu')

    # 加载训练好的学生模型
    model = load_student_model(device)
    model = model.cpu().eval()

    params = count_parameters(model)
    print(f"模型参数量: {params:,}")

    # 导出 ONNX
    os.makedirs('./models', exist_ok=True)
    onnx_path = './models/student_model.onnx'
    export_onnx(model, onnx_path, input_size=(1, 3, 224, 224))

    # 验证 ONNX 模型（可选）
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过 (格式正确)")
    except ImportError:
        print("提示: 安装 onnx 包可验证模型格式 (pip install onnx)")
    except Exception as e:
        print(f"ONNX 验证警告: {e}")

    print("\n下一步: 使用 ATC 工具将 ONNX 转换为 OM 模型")
    print("  运行: bash 06_atc_convert.sh")

    return onnx_path


if __name__ == '__main__':
    main()
