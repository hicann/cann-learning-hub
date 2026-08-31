# MNIST 数据集说明

本目录用于存放手写数字数据集 `mnist.npz`（约 11 MB）。

## 获取方式

MNIST 是公开数据集，**不再随仓库分发压缩包**，可通过以下任一方式获取 `mnist.npz`：

1. **直连下载（推荐）**：

   ```bash
   wget -O mnist.npz https://www.qmpan.com/f/Ek8AF3/mnist.npz
   ```

2. **自动下载**：直接运行本 Lab 的 notebook，代码会检测本目录下是否存在 `mnist.npz`，若不存在则优先从直连地址下载，失败再回退至官方源：

   ```
   https://www.qmpan.com/f/Ek8AF3/mnist.npz
   https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
   ```

3. **通过 TensorFlow/Keras 获取**：

   ```python
   import tensorflow as tf
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
   ```

下载完成后将 `mnist.npz` 放入本目录即可，notebook 会优先加载本地文件以节省时间。
