"""模型定义模块。

这里仍然使用适合作业讲解的 CNN，
但比最初的三层卷积版本更强一些：
1. 支持 batch / group / none 三种归一化；
2. 适度增加通道数；
3. 使用全局平均池化，避免把输入尺寸写死在 32x32。
"""

from __future__ import annotations

import mindspore.nn as nn
import mindspore.ops as ops


SUPPORTED_NORM_TYPES = ("group", "batch", "none")
SUPPORTED_POOL_TYPES = ("adaptive", "reduce_mean")


class Identity(nn.Cell):
    """兼容不同 MindSpore 版本的恒等映射。"""

    def construct(self, x):
        return x


class ReduceMeanGlobalAvgPool2d(nn.Cell):
    """Global average pooling implemented with ReduceMean for Ascend stability."""

    def __init__(self) -> None:
        super().__init__()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        return self.reduce_mean(x, (2, 3))


def pick_group_count(num_channels: int) -> int:
    """给 GroupNorm 选择一个稳定的分组数。"""

    for group_count in (16, 8, 4, 2, 1):
        if num_channels % group_count == 0:
            return group_count
    return 1


def make_conv_norm(num_channels: int, norm_type: str) -> nn.Cell:
    """为卷积特征图创建归一化层。"""

    normalized_type = str(norm_type).lower().strip()
    if normalized_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if normalized_type == "group":
        return nn.GroupNorm(pick_group_count(num_channels), num_channels)
    if normalized_type == "none":
        return Identity()
    raise ValueError(f"不支持的 norm_type: {norm_type}，可选值: {SUPPORTED_NORM_TYPES}")


def make_dense_norm(num_features: int, norm_type: str) -> nn.Cell:
    """为全连接层输出创建归一化层。"""

    normalized_type = str(norm_type).lower().strip()
    if normalized_type == "batch":
        return nn.BatchNorm1d(num_features)
    if normalized_type == "group":
        return nn.LayerNorm((num_features,))
    if normalized_type == "none":
        return Identity()
    raise ValueError(f"不支持的 norm_type: {norm_type}，可选值: {SUPPORTED_NORM_TYPES}")


class ConvBNReLU(nn.Cell):
    """一个基础卷积块：卷积 + 归一化 + ReLU。"""

    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "group") -> None:
        super().__init__()
        self.block = nn.SequentialCell(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode="pad", padding=1, has_bias=False),
                make_conv_norm(out_channels, norm_type),
                nn.ReLU(),
            ]
        )

    def construct(self, x):
        return self.block(x)


class SimpleCNN(nn.Cell):
    """一个适合 GTSRB 的中小型 CNN。

    设计思路：
    1. 结构清晰，方便初学者理解；
    2. 比最初版本更深一些，但远没有复杂到 ResNet 的程度；
    3. 通过全局平均池化兼容 32x32、48x48、64x64 等输入尺寸。
    """

    def __init__(
        self,
        num_classes: int = 43,
        input_channels: int = 3,
        norm_type: str = "group",
        pool_type: str = "adaptive",
    ) -> None:
        super().__init__()
        self.norm_type = str(norm_type).lower().strip()
        if self.norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(f"不支持的 norm_type: {norm_type}，可选值: {SUPPORTED_NORM_TYPES}")
        self.pool_type = str(pool_type).lower().strip()
        if self.pool_type not in SUPPORTED_POOL_TYPES:
            raise ValueError(f"不支持的 pool_type: {pool_type}，可选值: {SUPPORTED_POOL_TYPES}")

        # 每个阶段使用两个小卷积，再做一次池化。
        # 这种写法比“单层卷积直接池化”更容易学习到边缘、纹理和局部形状。
        self.features = nn.SequentialCell(
            [
                ConvBNReLU(input_channels, 32, norm_type=self.norm_type),
                ConvBNReLU(32, 32, norm_type=self.norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBNReLU(32, 64, norm_type=self.norm_type),
                ConvBNReLU(64, 64, norm_type=self.norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBNReLU(64, 128, norm_type=self.norm_type),
                ConvBNReLU(128, 128, norm_type=self.norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBNReLU(128, 256, norm_type=self.norm_type),
                ConvBNReLU(256, 256, norm_type=self.norm_type),
            ]
        )

        # 实验4默认沿用原始 AdaptiveAvgPool2d；实验5实时检测可显式切换 ReduceMean 避开 Ascend 算子问题。
        if self.pool_type == "reduce_mean":
            self.pool = ReduceMeanGlobalAvgPool2d()
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.SequentialCell(
            [
                nn.Flatten(),
                nn.Dense(256, 128),
                make_dense_norm(128, self.norm_type),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Dense(128, num_classes),
            ]
        )

    def construct(self, x):
        """定义前向传播。"""

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def create_model(
    num_classes: int = 43,
    input_channels: int = 3,
    norm_type: str = "group",
    pool_type: str = "adaptive",
) -> SimpleCNN:
    """统一的模型创建入口。"""

    return SimpleCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        norm_type=norm_type,
        pool_type=pool_type,
    )
