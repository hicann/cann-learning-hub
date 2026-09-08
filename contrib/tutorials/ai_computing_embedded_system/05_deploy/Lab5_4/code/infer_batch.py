"""
批量手写数字图片识别脚本
功能：加载 OM 模型，对 output/test_images/ 目录下所有测试图片进行批量识别，
      打印每张图片的识别结果，统计识别成功率，并生成汇总报告
运行环境：香橙派开发板（Ascend 310B4）

使用方法：
    python3 infer_batch.py [om_path] [test_dir]

示例：
    python3 infer_batch.py ../output/simplecnn_mnist_fp32.om \
                            ../output/test_images/
"""

import sys
import os
import time
import glob
import json
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acl_classifier import ACLClassifier
from preprocess import preprocess_image, parse_label_from_filename


def infer_batch(om_path, test_dir, verbose=True):
    """
    对目录下所有测试图片执行批量 OM 模型推理

    参数:
        om_path: OM 模型文件路径
        test_dir: 测试图片目录
        verbose: 是否打印详细信息
    返回:
        results: 识别结果列表，每个元素为 dict:
            {file, true, pred, correct, confidence, infer_time}
        accuracy: 识别成功率（百分比）
    """
    # 查找所有测试图片（排除结果图和汇总图）
    image_files = sorted([
        f for f in glob.glob(os.path.join(test_dir, 'test_*_label*.png'))
        if 'result' not in os.path.basename(f) and 'summary' not in os.path.basename(f)
    ])

    if not image_files:
        print(f"[ERROR] 在 {test_dir} 中未找到测试图片")
        return [], 0.0

    # 初始化 ACL 分类器并加载 OM 模型（只加载一次，复用于所有图片）
    classifier = ACLClassifier(om_path)
    classifier.init()
    classifier.load_model()

    results = []
    total_time = 0.0

    if verbose:
        print()
        print("=" * 65)
        print(f"批量数字图片识别 (共 {len(image_files)} 张)")
        print(f"OM 模型: {om_path}")
        print("=" * 65)
        print(f"{'序号':<6}{'图片文件':<28}{'实际数字':<10}{'识别数字':<10}{'结果':<8}{'耗时(ms)':<10}")
        print("-" * 65)

    for i, fpath in enumerate(image_files):
        # 预处理
        x = preprocess_image(fpath)

        # 推理并计时
        output, infer_time = classifier.infer(x)
        total_time += infer_time

        # 预测结果
        pred = int(output.argmax())
        exp_out = np.exp(output - output.max())
        probs = exp_out / exp_out.sum()
        confidence = float(probs[pred])

        # 真实标签
        true_label = parse_label_from_filename(fpath)
        correct = (pred == true_label) if true_label is not None else None

        results.append({
            'file': os.path.basename(fpath),
            'true': true_label,
            'pred': pred,
            'correct': correct,
            'confidence': confidence,
            'infer_time': infer_time,
        })

        if verbose:
            status = "正确" if correct else "错误" if correct is not None else "未知"
            print(f"{i+1:<6}{os.path.basename(fpath):<28}"
                  f"{str(true_label):<10}{pred:<10}{status:<8}{infer_time:<10.2f}")

    # 释放资源
    classifier.release()

    # 统计识别成功率
    valid = [r for r in results if r['correct'] is not None]
    num_correct = sum(1 for r in valid if r['correct'])
    num_valid = len(valid)
    accuracy = 100.0 * num_correct / num_valid if num_valid > 0 else 0.0
    avg_time = total_time / len(results) if results else 0.0

    if verbose:
        print("-" * 65)
        print(f"识别成功率: {num_correct}/{num_valid} = {accuracy:.1f}%")
        print(f"总耗时:     {total_time:.2f} ms")
        print(f"平均耗时:   {avg_time:.2f} ms/张")
        print(f"吞吐率:     {1000 / avg_time:.1f} 张/秒" if avg_time > 0 else "")
        print("=" * 65)

        # 按数字类别统计识别情况
        print()
        print("按数字类别统计:")
        print(f"{'数字':<8}{'总数':<8}{'正确':<8}{'错误':<8}{'成功率':<10}")
        print("-" * 42)
        for digit in range(10):
            digit_results = [r for r in valid if r['true'] == digit]
            d_total = len(digit_results)
            d_correct = sum(1 for r in digit_results if r['correct'])
            d_acc = 100.0 * d_correct / d_total if d_total > 0 else 0.0
            print(f"{digit:<8}{d_total:<8}{d_correct:<8}"
                  f"{d_total - d_correct:<8}{d_acc:<10.1f}%")
        print("=" * 42)

        # 列出识别错误的图片
        wrong = [r for r in valid if not r['correct']]
        if wrong:
            print()
            print(f"识别错误的图片 ({len(wrong)} 张):")
            for r in wrong:
                print(f"  {r['file']}: 实际={r['true']}, 识别={r['pred']}, "
                      f"置信度={r['confidence'] * 100:.1f}%")
        else:
            print()
            print("[OK] 所有图片识别正确！")

    # 保存结果到 JSON 文件
    result_json = os.path.join(test_dir, 'batch_result.json')
    with open(result_json, 'w') as f:
        json.dump({
            'om_path': om_path,
            'total': len(results),
            'correct': num_correct,
            'accuracy': accuracy,
            'avg_time_ms': avg_time,
            'results': results,
        }, f, indent=2)
    if verbose:
        print(f"\n结果已保存: {result_json}")

    return results, accuracy


if __name__ == '__main__':
    om_path = sys.argv[1] if len(sys.argv) > 1 else '../output/simplecnn_mnist_fp32.om'
    test_dir = sys.argv[2] if len(sys.argv) > 2 else '../output/test_images/'

    if not os.path.exists(om_path):
        print(f"[ERROR] OM 模型文件不存在: {om_path}")
        sys.exit(1)
    if not os.path.isdir(test_dir):
        print(f"[ERROR] 测试图片目录不存在: {test_dir}")
        sys.exit(1)

    infer_batch(om_path, test_dir)
