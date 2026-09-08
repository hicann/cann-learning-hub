"""
猫狗分类深度学习案例 - 预训练 ResNet18 推理
功能：使用预训练 ResNet18 模型对 4 张图片进行猫狗分类，并可视化结果
适用平台：香橙派 AIpro（昇腾 310B4）/ Atlas 910B3 / 通用 CPU 环境
运行方式：python3 cat_dog_classification.py

依赖包：torch, torchvision, matplotlib, Pillow
若未安装，建议执行：
    pip install torch torchvision matplotlib Pillow

说明：
    - 使用 ImageNet 预训练的 ResNet18 模型，无需训练，直接推理
    - ImageNet 中猫类别编号 281-285，狗类别编号 151-268
    - 图表标签使用英文，避免中文乱码
    - 结果保存到 output/ 目录
"""

import os
import sys

# 确保工作目录为脚本所在目录的上一级（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_model():
    """加载预训练 ResNet18 模型"""
    print("Loading pre-trained ResNet18 model...")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)  # 兼容旧版本
    model.eval()
    print("Model loaded!\n")
    return model


def get_transform():
    """定义图片预处理流程"""
    transform = transforms.Compose([
        transforms.Resize(256),                    # 缩放短边至 256
        transforms.CenterCrop(224),                # 中心裁剪为 224x224
        transforms.ToTensor(),                     # 转为张量，归一化到 0~1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def classify_image(model, transform, image_path, image_name):
    """对单张图片进行猫狗分类"""
    # 打开并预处理图片
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)
    prob = torch.softmax(output, dim=1)[0]

    # 计算猫和狗的置信度
    cat_conf = prob[281:286].sum().item()   # ImageNet 猫类别: 281-285
    dog_conf = prob[151:269].sum().item()   # ImageNet 狗类别: 151-268
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

    # 在柱子上方标注预测结果
    for i, pred in enumerate(preds):
        y_pos = max(cat_confs[i], dog_confs[i]) + 0.02
        ax.text(i, y_pos, f'-> {pred}', ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Cat vs Dog Classification Results', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/classification_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved to output/classification_result.png")


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

    plt.suptitle('Classification Results with Pre-trained ResNet18', fontsize=15)
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/prediction_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Summary saved to output/prediction_summary.png")


def main():
    print("=" * 60)
    print("Cat and Dog Classification with Pre-trained ResNet18")
    print("=" * 60)
    print()

    # 图片路径
    image_files = ['images/cat1.jpg', 'images/cat2.jpg',
                   'images/dog1.jpg', 'images/dog2.jpg']
    image_names = ['cat1', 'cat2', 'dog1', 'dog2']

    # 第 1 步：查看图片
    print("--- Step 1: Display images ---")
    show_images(image_files, image_names)
    print()

    # 第 2 步：加载模型并分类
    print("--- Step 2: Load model and classify ---")
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
