"""
utils.py - 公共工具函数模块
实验4.5 昇腾香橙派部署深度学习网络实验

本模块提供数据加载、模型评估、可视化等公共工具函数，
供 01_train_mobilenetv2.py ~ 07_acl_inference.py 共用。
"""

import os
import io
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


# ============================================================
# 1. 设备检测
# ============================================================
def get_device():
    """检测并返回计算设备：优先昇腾NPU，其次CUDA，最后CPU"""
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device('npu'), True
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda'), False
    return torch.device('cpu'), False


# ============================================================
# 2. 数据集与数据加载
# ============================================================
def resolve_image_dir(image_dir='./images'):
    """解析图片目录路径

    脚本可能从 code/ 目录运行，而图片位于上级 images/ 目录。
    依次尝试候选路径，返回第一个包含全部猫狗图片的目录；
    若均不存在则返回原始 image_dir（后续数据集为空时会给出清晰报错）。
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        image_dir,
        os.path.join(_here, 'images'),
        os.path.join(_here, '..', 'images'),
        os.path.join(os.getcwd(), 'images'),
        os.path.join(os.getcwd(), '..', 'images'),
    ]
    expected = {'cat1.jpg', 'cat2.jpg', 'dog1.jpg', 'dog2.jpg'}
    seen = set()
    for cand in candidates:
        cand = os.path.normpath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isdir(cand) and expected.issubset(set(os.listdir(cand))):
            return cand
    return image_dir


class CatDogDataset(Dataset):
    """猫狗分类数据集

    使用 images 目录下的 4 张图片：
      cat1.jpg, cat2.jpg -> 标签 0 (猫)
      dog1.jpg, dog2.jpg -> 标签 1 (狗)

    通过数据增强扩充训练集。
    """

    # 期望的图片文件名 -> 标签
    FILE_LABEL_MAP = {
        'cat1.jpg': 0, 'cat2.jpg': 0,
        'dog1.jpg': 1, 'dog2.jpg': 1
    }

    def __init__(self, image_dir='./images', augment_times=1, transform=None):
        self.transform = transform
        self.image_dir = resolve_image_dir(image_dir)
        self.samples = []
        for fname, label in self.FILE_LABEL_MAP.items():
            path = os.path.join(self.image_dir, fname)
            if os.path.exists(path):
                for _ in range(augment_times):
                    self.samples.append((path, label))
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"在目录 '{self.image_dir}' 中未找到猫狗图片 "
                f"(期望: {sorted(self.FILE_LABEL_MAP.keys())})。"
                f"请从 Lab4_5 根目录或 code/ 目录运行脚本，"
                f"或将 images/ 目录放在脚本可访问的位置。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    """返回训练和测试的数据预处理流水线"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def get_dataloaders(image_dir='./images', augment_times=50, batch_size=16):
    """创建训练和测试 DataLoader"""
    train_transform, test_transform = get_transforms()
    train_dataset = CatDogDataset(image_dir, augment_times, train_transform)
    test_dataset = CatDogDataset(image_dir, 1, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset


# ============================================================
# 3. 模型评估工具
# ============================================================
def count_parameters(model):
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """计算模型 state_dict 的存储大小（MB）"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024 / 1024


def evaluate_model(model, test_loader, device):
    """在测试集上评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total


def measure_inference_time(model, test_loader, device, num_runs=20):
    """测量模型平均推理时间（ms）"""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            for images, _ in test_loader:
                images = images.to(device)
                if device.type == 'npu':
                    torch.npu.synchronize()
                start = time.time()
                _ = model(images)
                if device.type == 'npu':
                    torch.npu.synchronize()
                times.append(time.time() - start)
    return np.mean(times) * 1000


# ============================================================
# 4. 训练函数
# ============================================================
def train_model(model, train_loader, test_loader, device,
                num_epochs=5, learning_rate=0.001, model_name='Model'):
    """训练模型并返回训练历史记录"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'training_time': 0.0
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            running_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        test_acc = evaluate_model(model, test_loader, device)
        test_loss = 0.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)

        print(f'  [{model_name}] Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss={train_loss:.4f} Acc={train_acc:.4f} | '
              f'Test Acc={test_acc:.4f}')

    elapsed = time.time() - start_time
    history['training_time'] = elapsed
    print(f'  [{model_name}] Training completed in {elapsed:.2f}s')
    return history


# ============================================================
# 5. 稀疏率检查（用于裁剪实验）
# ============================================================
def check_sparsity(model):
    """统计模型中 Conv2d 和 Linear 层的权重稀疏率"""
    total = 0
    zero = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            total += w.numel()
            zero += (w == 0).sum().item()
    sparsity = 100.0 * zero / total if total > 0 else 0.0
    print(f'稀疏率 Sparsity: {sparsity:.2f}%  (零值 {zero} / 总计 {total})')
    return sparsity


# ============================================================
# 6. 量化后端选择（用于量化实验）
# ============================================================
def setup_quant_engine():
    """根据 CPU 架构自动选择量化后端：x86 -> fbgemm, ARM -> qnnpack"""
    import platform
    machine = platform.machine().lower()
    supported = torch.backends.quantized.supported_engines
    if machine in ("x86_64", "amd64") and "fbgemm" in supported:
        engine = "fbgemm"
    elif "qnnpack" in supported:
        engine = "qnnpack"
    else:
        engine = supported[0]
    torch.backends.quantized.engine = engine
    return engine


if __name__ == '__main__':
    device, has_npu = get_device()
    print(f"Device: {device}, NPU available: {has_npu}")
    print(f"PyTorch: {torch.__version__}")
