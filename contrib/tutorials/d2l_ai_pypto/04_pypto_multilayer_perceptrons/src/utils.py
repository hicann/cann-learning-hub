"""纯 PyTorch 公共工具模块（替代 d2l 包）。

本模块提供 d2l 常用函数的纯 PyTorch 等价实现，供第 4 章答案 notebook
通过 ``from src.utils import ...`` 复用，避免依赖 d2l 包。

导出清单：
- 数据加载：load_data_fashion_mnist, load_array, synthetic_data
- 训练/评估：train_epoch_ch3, train_ch3, evaluate_accuracy, evaluate_loss
- 可视化：Animator, plot, set_axes
"""

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_data_fashion_mnist(batch_size, resize=None, root='../data',
                            num_workers=0):
    """下载并加载 Fashion-MNIST 数据集。

    Args:
        batch_size: 批量大小
        resize: 如果指定，将图像调整为该尺寸
        root: 数据存储根目录
        num_workers: DataLoader 工作进程数

    Returns:
        (train_iter, test_iter): 训练和测试数据迭代器
    """
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trans, download=True)
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False,
                           num_workers=num_workers)
    return train_iter, test_iter

def load_array(data_arrays, batch_size, is_train=True):
    """构造 PyTorch 数据迭代器。

    Args:
        data_arrays: (features, labels) 元组或列表
        batch_size: 批量大小
        is_train: True 则打乱数据

    Returns:
        DataLoader 迭代器
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

def synthetic_data(w, b, num_examples, device=None):
    """生成 y = Xw + b + noise 的合成数据。

    Args:
        w: 权重向量
        b: 偏置标量
        num_examples: 样本数
        device: 张量所在设备（None 表示 CPU）

    Returns:
        (X, y): 特征和标签
    """
    X = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape, device=device)
    return X, y.reshape((-1, 1))

# ── 内部辅助 ──────────────────────────────────────────────────────────────

class Accumulator:
    """在 n 个变量上累加求和。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def _accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# ── 训练/评估 ─────────────────────────────────────────────────────────────

def evaluate_accuracy(net, data_iter):
    """在指定数据集上计算模型准确率。

    Args:
        net: 模型
        data_iter: 数据迭代器

    Returns:
        准确率（0~1 之间的浮点数）
    """
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def evaluate_loss(net, data_iter, loss):
    """在指定数据集上计算平均损失。

    Args:
        net: 模型
        data_iter: 数据迭代器
        loss: 损失函数

    Returns:
        平均损失值
    """
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（第 3 章风格）。

    Args:
        net: 模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 优化器（torch.optim.Optimizer）或自定义更新函数

    Returns:
        (train_loss, train_acc): 平均训练损失和准确率
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), _accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型并可视化（第 3 章风格）。

    Args:
        net: 模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss: 损失函数
        num_epochs: 训练轮数
        updater: 优化器
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# ── 可视化 ────────────────────────────────────────────────────────────────

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置 matplotlib 坐标轴的属性。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def use_svg_display():
    """使用 SVG 格式在 notebook 中显示图形。"""
    try:
        import matplotlib_inline
        matplotlib_inline.backend_inline.set_matplotlib_formats('png')
    except (ImportError, AttributeError):
        pass

class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """向图表中添加多个数据点。"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。"""
    if legend is None:
        legend = []

    def _set_axes(ax):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=figsize)
    axes.cla()
    if Y is not None:
        if isinstance(Y, (list, tuple)):
            Y = [y.detach().cpu().numpy() if torch.is_tensor(y) else y for y in Y]
        elif torch.is_tensor(Y):
            Y = Y.detach().cpu().numpy()
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        axes.plot(X, *Y) if isinstance(Y, list) else axes.plot(X, Y)
    else:
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        axes.plot(X)
    _set_axes(axes)

# ── 从零实现辅助函数 ─────────────────────────────────────────────────────

def sgd(params, lr, batch_size):
    """小批量随机梯度下降（手写版，含 /batch_size 归一化）。

    Args:
        params: 参数列表
        lr: 学习率
        batch_size: 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def get_fashion_mnist_labels(labels):
    """将 Fashion-MNIST 类别索引转换为文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像网格。"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.squeeze().detach().numpy(), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

