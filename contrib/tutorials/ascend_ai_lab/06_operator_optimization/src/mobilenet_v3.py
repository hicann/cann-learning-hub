# ====== MobileNetV3 完整模型定义（内嵌，无需外部文件）======
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== MobileNetV3 完整模型定义（自包含，不依赖外部文件）======

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x

class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )
    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out).view(batch, channels, 1, 1)
        return out * x

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super().__init__()
        padding = (kernal_size - 1) // 2
        self.use_connect = stride == 1 and in_channels == out_channels
        self.SE = SE
        activation = nn.ReLU if nonLinear == "RE" else h_swish
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp_size), activation(inplace=True))
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernal_size, stride, padding, groups=exp_size),
            nn.BatchNorm2d(exp_size))
        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)
        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels), activation(inplace=True))
    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.SE:
            out = self.squeeze_block(out)
        out = self.point_conv(out)
        return x + out if self.use_connect else out

class MobileNetV3(nn.Module):
    def __init__(self, model_mode="LARGE", num_classes=1000, multiplier=1.0, dropout_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        if model_mode == "LARGE":
            layers = [
                [16, 16, 3, 1, "RE", False, 16], [16, 24, 3, 2, "RE", False, 64],
                [24, 24, 3, 1, "RE", False, 72], [24, 40, 5, 2, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120], [40, 40, 5, 1, "RE", True, 120],
                [40, 80, 3, 2, "HS", False, 240], [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184], [80, 80, 3, 1, "HS", False, 184],
                [80, 112, 3, 1, "HS", True, 480], [112, 112, 3, 1, "HS", True, 672],
                [112, 160, 5, 1, "HS", True, 672], [160, 160, 5, 2, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960],
            ]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(
                nn.Conv2d(3, init_conv_out, 3, 2, 1), nn.BatchNorm2d(init_conv_out), h_swish())
            self.block = nn.Sequential(*[
                MobileBlock(_make_divisible(ic * multiplier), _make_divisible(oc * multiplier),
                           k, s, nl, se, _make_divisible(exp * multiplier))
                for ic, oc, k, s, nl, se, exp in layers])
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(_make_divisible(160 * multiplier), _make_divisible(960 * multiplier), 1, 1),
                nn.BatchNorm2d(_make_divisible(960 * multiplier)), h_swish())
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(_make_divisible(960 * multiplier), _make_divisible(1280 * multiplier), 1, 1),
                h_swish(), nn.Dropout(dropout_rate),
                nn.Conv2d(_make_divisible(1280 * multiplier), num_classes, 1, 1))
        elif model_mode == "SMALL":
            layers = [
                [16, 16, 3, 2, "RE", True, 16], [16, 24, 3, 2, "RE", False, 72],
                [24, 24, 3, 1, "RE", False, 88], [24, 40, 5, 2, "RE", True, 96],
                [40, 40, 5, 1, "RE", True, 240], [40, 40, 5, 1, "RE", True, 240],
                [40, 48, 5, 1, "HS", True, 120], [48, 48, 5, 1, "HS", True, 144],
                [48, 96, 5, 2, "HS", True, 288], [96, 96, 5, 1, "HS", True, 576],
                [96, 96, 5, 1, "HS", True, 576],
            ]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(
                nn.Conv2d(3, init_conv_out, 3, 2, 1), nn.BatchNorm2d(init_conv_out), h_swish())
            self.block = nn.Sequential(*[
                MobileBlock(_make_divisible(ic * multiplier), _make_divisible(oc * multiplier),
                           k, s, nl, se, _make_divisible(exp * multiplier))
                for ic, oc, k, s, nl, se, exp in layers])
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(_make_divisible(96 * multiplier), _make_divisible(576 * multiplier), 1, 1),
                SqueezeBlock(_make_divisible(576 * multiplier)),
                nn.BatchNorm2d(_make_divisible(576 * multiplier)), h_swish())
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(_make_divisible(576 * multiplier), _make_divisible(1280 * multiplier), 1, 1),
                h_swish(), nn.Dropout(dropout_rate),
                nn.Conv2d(_make_divisible(1280 * multiplier), num_classes, 1, 1))
        self.apply(_weights_init)
    def forward(self, x):
        out = self.block(self.init_conv(x))
        out = self.out_conv1(out)
        b, c, h, w = out.size()
        return self.out_conv2(F.avg_pool2d(out, [h, w])).view(b, -1)
