"""
图像预处理模块
功能：将手写数字图片预处理为模型推理所需的输入格式
运行环境：香橙派开发板

预处理流程：
    1. 读取图片并转为灰度
    2. 缩放到 28x28（MNIST 标准尺寸）
    3. 像素值归一化到 [0, 1]
    4. 使用 MNIST 均值/标准差标准化
    5. 调整为 NCHW 格式 (1, 1, 28, 28)

使用示例：
    from preprocess import preprocess_image, load_image
    x = preprocess_image('output/test_images/test_00_label0.png')
"""

import numpy as np

# MNIST 标准化参数（与训练时一致）
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_image(image_path):
    """
    加载图片为 numpy 灰度数组

    参数:
        image_path: 图片文件路径
    返回:
        img: numpy 数组, shape=(28, 28), dtype=uint8
    """
    from PIL import Image
    img = Image.open(image_path).convert('L')
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.BILINEAR)
    return np.array(img)


def preprocess_image(image):
    """
    将图片预处理为模型输入格式

    参数:
        image: 可以是图片路径(str)或 numpy 数组(shape=(28,28)或(H,W))
    返回:
        x: numpy 数组, shape=(1, 1, 28, 28), dtype=float32
    """
    if isinstance(image, str):
        image = load_image(image)

    image = image.astype(np.float32)

    # 像素值归一化到 [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # 使用 MNIST 均值/标准差标准化（与训练时一致）
    image = (image - MNIST_MEAN) / MNIST_STD

    # 调整为 NCHW 格式: (1, 1, 28, 28)
    return image.reshape(1, 1, 28, 28)


def parse_label_from_filename(filename):
    """
    从文件名中解析真实标签
    文件名格式: test_XX_labelY.png -> 返回 Y

    参数:
        filename: 文件名（如 test_05_label3.png）
    返回:
        label: 真实标签（int），无法解析时返回 None
    """
    import os
    basename = os.path.basename(filename)
    if 'label' in basename:
        try:
            return int(basename.split('label')[1].split('.')[0])
        except (ValueError, IndexError):
            return None
    return None
