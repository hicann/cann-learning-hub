"""
猫狗分类深度学习案例 - NPU 加速推理版本
功能：使用预训练 ResNet18 模型在昇腾 NPU 上进行猫狗分类推理
适用平台：香橙派 AIpro（昇腾 310B4）/ Atlas 910B3
运行方式：python3 cat_dog_classification_npu.py

依赖包：torch, torchvision, torch_npu, matplotlib, Pillow
若未安装，建议执行：
    pip install torch torchvision torch_npu matplotlib Pillow

说明：
    - 与 cat_dog_classification.py 功能一致，但使用 NPU 加速推理
    - 通过 .npu() 将模型和数据搬到昇腾 NPU 上执行
    - 适合在香橙派等搭载昇腾 NPU 的设备上运行
"""

import os
import sys
import ctypes
import ctypes.util

# 确保工作目录为脚本所在目录的上一级（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# 预加载 libgomp，避免 "cannot allocate memory in static TLS block" 错误
try:
    _gomp = ctypes.util.find_library('gomp')
    if _gomp:
        ctypes.CDLL(_gomp, mode=ctypes.RTLD_GLOBAL)
    else:
        for _p in ['/usr/lib/aarch64-linux-gnu/libgomp.so.1',
                    '/lib64/libgomp.so.1',
                    '/usr/lib64/libgomp.so.1']:
            if os.path.exists(_p):
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
                break
except Exception:
    pass

import torch
import torch_npu
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_model():
    """加载预训练 ResNet18 模型并搬到 NPU"""
    print("Loading pre-trained ResNet18 model...")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.eval()
    # 将模型搬到 NPU，并转为 float16（昇腾 310B4 的 MaxPool 等算子仅支持 float16）
    model = model.npu().half()
    print(f"Model loaded on device: {next(model.parameters()).device}")
    print()
    return model


def get_transform():
    """定义图片预处理流程"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def classify_image(model, transform, image_path, image_name):
    """对单张图片进行猫狗分类（NPU 推理）"""
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).npu().half()  # 数据搬到 NPU，并转为 float16

    with torch.no_grad():
        output = model(input_tensor)
    prob = torch.softmax(output, dim=1)[0]

    cat_conf = prob[281:286].sum().item()
    dog_conf = prob[151:269].sum().item()
    prediction = 'cat' if cat_conf > dog_conf else 'dog'

    result = {
        'name': image_name,
        'file': image_path,
        'prediction': prediction,
        'cat_conf': cat_conf,
        'dog_conf': dog_conf
    }

    print(f"{image_name}: predicted = {prediction:3s}  |  "
          f"cat_conf = {cat_conf:.4f}  |  dog_conf = {dog_conf:.4f}")
    return result


def show_images(image_files, titles):
    """显示 4 张原始图片"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, f, t in zip(axes, image_files, titles):
        img = Image.open(f)
        ax.imshow(img)
        ax.set_title(t, fontsize=14)
        ax.axis('off')
    plt.suptitle('Cat and Dog Images', fontsize=16)
    plt.tight_layout()
    plt.show()
    print("4 images loaded successfully!")


def plot_confidence_bar(results):
    """绘制分类置信度柱状图"""
    names = [r['name'] for r in results]
    cat_confs = [r['cat_conf'] for r in results]
    dog_confs = [r['dog_conf'] for r in results]
    preds = [r['prediction'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    bar_width = 0.35

    ax.bar([i - bar_width / 2 for i in x], cat_confs, bar_width,
           label='Cat Confidence', color='orange')
    ax.bar([i + bar_width / 2 for i in x], dog_confs, bar_width,
           label='Dog Confidence', color='steelblue')

    for i, pred in enumerate(preds):
        y_pos = max(cat_confs[i], dog_confs[i]) + 0.02
        ax.text(i, y_pos, f'-> {pred}', ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Cat vs Dog Classification Results (NPU)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/classification_result_npu.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved to output/classification_result_npu.png")


def plot_prediction_summary(results):
    """在图片上标注预测结果并显示"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = {'cat': 'orange', 'dog': 'steelblue'}

    for ax, r in zip(axes, results):
        img = Image.open(r['file'])
        ax.imshow(img)
        pred = r['prediction']
        conf = max(r['cat_conf'], r['dog_conf'])
        ax.set_title(f"{r['name']} -> {pred}\n(conf: {conf:.2%})",
                     fontsize=12, color=colors[pred], fontweight='bold')
        ax.axis('off')

    plt.suptitle('Classification Results with ResNet18 (NPU Accelerated)', fontsize=15)
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/prediction_summary_npu.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Summary saved to output/prediction_summary_npu.png")


def main():
    print("=" * 60)
    print("Cat and Dog Classification with ResNet18 (NPU Accelerated)")
    print("=" * 60)
    print()

    # 检查 NPU 可用性
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"NPU name: {torch.npu.get_device_name(0)}")
    print()

    image_files = ['images/cat1.jpg', 'images/cat2.jpg',
                   'images/dog1.jpg', 'images/dog2.jpg']
    image_names = ['cat1', 'cat2', 'dog1', 'dog2']

    # 第 1 步：查看图片
    print("--- Step 1: Display images ---")
    show_images(image_files, image_names)
    print()

    # 第 2 步：加载模型并分类
    print("--- Step 2: Load model and classify (NPU) ---")
    model = load_model()
    transform = get_transform()

    results = []
    for name, f in zip(image_names, image_files):
        result = classify_image(model, transform, f, name)
        results.append(result)

    print("\nClassification complete!")
    print()

    # 第 3 步：可视化分类结果
    print("--- Step 3: Visualize results ---")
    plot_confidence_bar(results)
    print()

    # 第 4 步：在图片上标注预测结果
    print("--- Step 4: Show predictions on images ---")
    plot_prediction_summary(results)

    print()
    print("=" * 60)
    print("Done! Results saved to output/ directory.")
    print("=" * 60)


if __name__ == '__main__':
    main()
