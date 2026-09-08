import numpy as np
from PIL import Image
import os

def preprocess_image(image_path, output_bin_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image, dtype=np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = (image / 255.0 - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    img_array = np.ascontiguousarray(image, dtype=np.float32)

    with open(output_bin_path, 'wb') as f:
        img_array.tofile(f)

    print(f"预处理完成，已保存到 {output_bin_path}")

# def preprocess_image(image_path, output_bin_path):
#     # 打开图像文件
#     with Image.open(image_path) as img:
#         # 转换为RGB模式（如果需要的话）
#         img = img.convert('RGB')
#         # 调整图像大小，例如：缩放到224x224
#         img = img.resize((224, 224))
#         # 将图像转换为NumPy数组
#         img_array = np.array(img)
#         # 归一化到[0, 1]范围
#         img_array = img_array / 255.0
#         # 可以添加更多的预处理步骤，例如数据类型转换等
#         img_array = img_array.astype('float32')  # 转换为float32类型
#         
#         # 将NumPy数组保存为二进制文件
#         with open(output_bin_path, 'wb') as f:
#             img_array.tofile(f)
# 
#     print(f"预处理完成，已保存到 {output_bin_path}")

def load_binary_image(bin_path, shape=(224, 224, 3)):
    # 从二进制文件加载数据
    with open(bin_path, 'rb') as f:
        img_array = np.fromfile(f, dtype=np.float32)

    # 重塑数组以匹配原始图像的形状
    img_array = img_array.reshape(shape)

    return img_array

if __name__ == '__main__':
    image_path = '/home/foxtrot024/my_mindspore/07_MyLab/lab_06/dog.jpg'
    output_bin_path = 'input_dog.bin'
    preprocess_image(image_path, output_bin_path)

    loaded_img = load_binary_image(output_bin_path)
    print("加载的图像数据形状:", loaded_img.shape)

    image_path = '/home/foxtrot024/my_mindspore/07_MyLab/lab_06/cat.jpg'
    output_bin_path = 'input_cat.bin'
    preprocess_image(image_path, output_bin_path)

    loaded_img = load_binary_image(output_bin_path)
    print("加载的图像数据形状:", loaded_img.shape)
