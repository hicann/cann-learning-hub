"""
单张手写数字图片识别脚本
功能：加载 OM 模型，对单张手写数字图片进行识别，打印识别结果
运行环境：香橙派开发板（Ascend 310B4）

使用方法：
    python3 infer_single.py <om_path> <image_path>

示例：
    python3 infer_single.py ../output/simplecnn_mnist_fp32.om \
                             ../output/test_images/test_00_label0.png
"""

import sys
import os
import time
import numpy as np

# 添加当前目录到路径，以便导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acl_classifier import ACLClassifier
from preprocess import preprocess_image, load_image, parse_label_from_filename


def infer_single(om_path, image_path, verbose=True):
    """
    对单张图片执行 OM 模型推理

    参数:
        om_path: OM 模型文件路径
        image_path: 待识别的图片路径
        verbose: 是否打印详细信息
    返回:
        pred: 预测的数字（0-9）
        true_label: 真实标签（从文件名解析，无法解析时为 None）
        confidence: 置信度（softmax 最大值）
        infer_time: 推理耗时（毫秒）
    """
    # 加载原始图片用于显示
    img = load_image(image_path)

    # 预处理
    x = preprocess_image(image_path)

    # 初始化 ACL 分类器并加载 OM 模型
    classifier = ACLClassifier(om_path)
    classifier.init()
    classifier.load_model()

    # 执行推理并计时
    output, infer_time = classifier.infer(x)

    # 计算预测结果与置信度
    pred = int(output.argmax())
    # softmax 计算置信度
    exp_out = np.exp(output - output.max())
    probs = exp_out / exp_out.sum()
    confidence = float(probs[pred])

    # 从文件名解析真实标签
    true_label = parse_label_from_filename(image_path)

    # 释放资源
    classifier.release()

    if verbose:
        print()
        print("=" * 55)
        print("单张数字图片识别结果")
        print("=" * 55)
        print(f"  OM 模型:     {om_path}")
        print(f"  图片文件:    {os.path.basename(image_path)}")
        print(f"  实际数字:    {true_label if true_label is not None else '未知'}")
        print(f"  识别数字:    {pred}")
        if true_label is not None:
            status = "正确" if pred == true_label else "错误"
            print(f"  识别结果:    {status}")
        print(f"  置信度:      {confidence * 100:.2f}%")
        print(f"  推理耗时:    {infer_time:.2f} ms")
        print("-" * 55)
        print("各类别概率分布:")
        for i in range(10):
            bar = "#" * int(probs[i] * 50)
            marker = " <-- 预测" if i == pred else ""
            print(f"  数字 {i}: {probs[i] * 100:6.2f}% |{bar}{marker}")
        print("=" * 55)

    return pred, true_label, confidence, infer_time


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("使用方法: python3 infer_single.py <om_path> <image_path>")
        print("示例:")
        print("  python3 infer_single.py ../output/simplecnn_mnist_fp32.om \\")
        print("                           ../output/test_images/test_00_label0.png")
        sys.exit(1)

    om_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(om_path):
        print(f"[ERROR] OM 模型文件不存在: {om_path}")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片文件不存在: {image_path}")
        sys.exit(1)

    infer_single(om_path, image_path)
