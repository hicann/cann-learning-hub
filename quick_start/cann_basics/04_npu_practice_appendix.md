# NPU 实践题附录：详细解读与知识点

> 本文件是 `04_npu_practice.ipynb` 的配套附录，对四道实践题的背景知识、详细答案、知识点展开说明。

---

## 目录

- [Hello NPU：环境检查与第一个 NPU 程序](#hello-npu环境检查与第一个-npu-程序)
- [第 1 题（基础）：在 NPU 上计算 z = x² + y](#第-1-题基础在-npu-上计算-z--x²--y)
- [第 2 题（进阶）：CPU vs NPU 矩阵乘法性能对比](#第-2-题进阶cpu-vs-npu-矩阵乘法性能对比)
- [第 3 题（挑战）：Batch 矩阵乘法 vs 循环单次矩阵乘法](#第-3-题挑战batch-矩阵乘法-vs-循环单次矩阵乘法)
- [第 4 题（挑战）：在 NPU 上加速图像高斯模糊](#第-4-题挑战在-npu-上加速图像高斯模糊)

---

## Hello NPU：环境检查与第一个 NPU 程序

### 背景知识

在开始实践题之前，需要确认 NPU 环境就绪。与 GPU 类似，PyTorch 通过设备后端来管理异构计算资源：

- **CPU（Host）**：运行 Python 代码、创建张量、调度任务
- **NPU（Device）**：执行张量计算，拥有独立的 HBM 内存

`torch_npu` 是昇腾 NPU 的 PyTorch 适配层，导入后会自动注册 `npu` 设备，使所有标准 PyTorch 操作可以无缝迁移到 NPU 执行。

### 代码解读

```python
import torch
import torch_npu

# ① 检查 NPU 是否可用
print(f"NPU 可用: {torch.npu.is_available()}")
print(f"NPU 设备: {torch.npu.get_device_name(0)}")

# ② 在 NPU 上做一个简单加法
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = (a.npu() + b.npu()).cpu()  # 搬到 NPU 计算，再搬回 CPU

print(f"a + b = {c.tolist()}  (Hello, we are running on NPU.)")
```

**关键三步**：

| 步骤 | API | 说明 |
|:-----|:----|:-----|
| 数据搬运 Host → Device | `tensor.npu()` | 将张量从 CPU 内存拷贝到 NPU HBM |
| NPU 计算 | `a + b` | 加法算子自动调度到 NPU 的 Vector 单元执行 |
| 结果搬运 Device → Host | `tensor.cpu()` | 将结果从 NPU HBM 拷回 CPU 内存 |

### 知识点：异构计算模型

```
CPU (Host)                          NPU (Device)
┌──────────────┐                   ┌──────────────────┐
│ Python 代码   │  ──.npu()──→     │ HBM (设备内存)     │
│ 创建张量 a, b │                   │ a_npu, b_npu      │
│              │                   │                   │
│ 等待结果      │  ←─.cpu()──      │ Vector 单元: a+b  │
│ c = result   │                   │                   │
└──────────────┘                   └──────────────────┘
```

CPU 负责"发任务、传数据、收结果"，NPU 负责"拼命算"。两者通过 `.npu()` / `.cpu()` 进行数据搬运。

---

## 第 1 题（基础）：在 NPU 上计算 z = x² + y

### 背景知识

**张量（Tensor）** 是深度学习中最基本的数据结构，由 `shape`（形状）和 `dtype`（数据类型）描述。本题使用一维张量（向量），形状为 `(1000,)`，即 1000 个元素的向量。

**FP32（单精度浮点数）** 是深度学习中最常用的数据类型，每个元素占 4 字节。NPU 的 Vector 计算单元针对 FP32/FP16 做了高度优化。

**逐元素运算（Element-wise Operation）** 是指对张量中每个元素独立计算，不涉及跨元素的数据依赖：
- 加法 `a + b`：`c[i] = a[i] + b[i]`
- 乘法 `a * b`：`c[i] = a[i] * b[i]`
- 平方 `x * x`：`c[i] = x[i] * x[i]`

这类运算由 NPU 的 **Vector 单元** 执行（而非 Cube 单元，Cube 专门做矩阵乘法）。

### 详细答案

```python
import torch
import torch_npu

# Step 1: 创建 x 和 y（CPU 上，FP32，形状 (1000,)）
x = torch.full((1000,), 2.0, dtype=torch.float32)
y = torch.full((1000,), 3.0, dtype=torch.float32)

# Step 2: 搬到 NPU
x_npu = x.npu()
y_npu = y.npu()

# Step 3: 在 NPU 上计算 z = x² + y
z_npu = x_npu * x_npu + y_npu

# Step 4: 搬回 CPU
z = z_npu.cpu()

# 验证
expected = torch.full((1000,), 7.0)
print(f"z 的前 5 个元素: {z[:5].tolist()}")
print(f"z 的设备: {z_npu.device}")
print(f"验证结果: {'✓ 通过' if torch.allclose(z, expected) else '✗ 失败'}")
```

**预期输出**：
```
z 的前 5 个元素: [7.0, 7.0, 7.0, 7.0, 7.0]
z 的设备: npu:0
验证结果: ✓ 通过
```

### 知识点详解

#### 1. `torch.full` vs `torch.ones` / `torch.zeros`

```python
torch.full((1000,), 2.0)   # 创建全为 2.0 的张量
torch.ones(1000) * 2.0      # 等价但多一次乘法运算
```

`torch.full` 直接填充指定值，效率更高。

#### 2. 计算图与算子

`z = x² + y` 对应的计算图：

```
x ──→ Square ──→ ──→ Add ──→ z
                    ↑
y ──────────────────┘
```

包含两个算子：
- **Square**（逐元素平方）：`x * x`，由 Vector 单元执行
- **Add**（逐元素加法）：`sq + y`，由 Vector 单元执行

#### 3. `torch.allclose` 验证

```python
torch.allclose(z, expected)  # 默认 atol=1e-8, rtol=1e-5
```

逐元素检查 `|z[i] - expected[i]| ≤ atol + rtol × |expected[i]|`。由于浮点数精度问题，很少用 `==` 判断相等，而是用 `allclose` 允许微小误差。

---

## 第 2 题（进阶）：CPU vs NPU 矩阵乘法性能对比

### 背景知识

**矩阵乘法（MatMul）** 是深度学习的核心运算。全连接层、注意力机制、卷积（im2col 变换后）底层都是矩阵乘法。

对于 `C = A × B`，其中 A 是 `M×K`，B 是 `K×N`，C 是 `M×N`：
- 计算量：`2×M×N×K` FLOPs（每个元素 K 次乘加）
- 数据量：`(M×K + K×N + M×N) × 4` Bytes（FP32）

**NPU 的 Cube 单元** 专为矩阵乘法设计，单时钟周期可完成 16×16（FP16）或类似规模的矩阵乘片段。大矩阵乘法是 NPU 最擅长的场景。

**异步执行**：NPU 计算是异步的——调用 `torch.matmul` 只是下发任务，不等待完成。要准确计时，必须用 `torch.npu.synchronize()` 等待所有 NPU 任务完成。

### 详细答案

```python
import time
import torch
import torch_npu

sizes = [128, 512, 2048]

print(f"{'规模':>8} | {'CPU 耗时':>12} | {'NPU 耗时':>12} | {'加速比':>8} | {'结果一致':>8}")
print("-" * 65)

for N in sizes:
    # Step 1: 创建两个 N×N 随机矩阵（CPU 上）
    a_cpu = torch.randn(N, N, dtype=torch.float32)
    b_cpu = torch.randn(N, N, dtype=torch.float32)

    # Step 2: CPU 矩阵乘法计时
    t0 = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    t_cpu = time.time() - t0

    # Step 3: NPU 矩阵乘法计时
    a_npu = a_cpu.npu()
    b_npu = b_cpu.npu()
    torch.npu.synchronize()  # 确保数据搬运完成
    t0 = time.time()
    c_npu = torch.matmul(a_npu, b_npu)
    torch.npu.synchronize()  # 等待计算完成
    t_npu = time.time() - t0

    # Step 4: 验证结果一致
    consistent = torch.allclose(c_cpu, c_npu.cpu(), atol=1e-1)

    # 打印结果
    speedup = t_cpu / t_npu
    print(f"{N:>8} | {t_cpu*1000:>10.2f}ms | {t_npu*1000:>10.2f}ms | {speedup:>7.1f}x | {'✓' if consistent else '✗':>8}")
```

**预期输出**（具体数值因环境而异）：
```
     128 |       0.51ms |       0.23ms |     2.2x |        ✓
     512 |       1.32ms |       0.21ms |     6.3x |        ✓
    2048 |     128.16ms |       0.44ms |   289.8x |        ✓
```

### 知识点详解

#### 1. 为什么 N=128 时加速比不高？

矩阵越小，数据搬运开销占比越大。N=128 的矩阵只有 128×128×4 = 64KB，搬运到 NPU 的时间可能比计算时间还长。

**算术强度**（Arithmetic Intensity）= FLOPs / Bytes：

**FLOPs 计算**（矩阵乘法 `C = A × B`，A: `N×N`，B: `N×N`，C: `N×N`）：
- 每个输出元素 `C[i][j]` 需要 N 次乘法和 N 次加法 = `2N` FLOPs
- 共 `N×N` 个输出元素
- 总 FLOPs = `2N × N × N = 2N³`

**Bytes 计算**（FP32，每个元素 4 字节）：
- 读矩阵 A：`N×N×4` 字节
- 读矩阵 B：`N×N×4` 字节
- 写矩阵 C：`N×N×4` 字节
- 总 Bytes = `3 × N² × 4`

**算术强度** = `2N³ / (3 × N² × 4)` = `N / 6`

| N | FLOPs (`2N³`) | Bytes (`3N²×4`) | 算术强度 (`N/6`) | 瓶颈 |
|---|-------|-------|---------|------|
| 128 | 4M | 196KB | 21 | 偏访存 |
| 512 | 268M | 3MB | 85 | 计算 |
| 2048 | 17G | 50MB | 341 | 计算 |

算术强度越高，NPU 算力优势越能充分发挥。

#### 2. `atol=1e-1` 容差的必要性

CPU 和 NPU 使用不同的数学库（CPU 用 OpenBLAS/MKL，NPU 用 CANN 算子库），浮点运算顺序不同，结果有微小差异。`1e-1` 的绝对容差对于随机矩阵乘法是合理的。

#### 3. NPU 首次执行的编译开销

NPU 首次执行某个算子时，CANN 需要编译生成 Kernel 指令（JIT 编译）。这会导致首次计时偏高。如果想测纯计算性能，可以先做一次 warmup：

```python
# Warmup（不计入计时）
_ = torch.matmul(a_npu, b_npu)
torch.npu.synchronize()

# 正式计时
t0 = time.time()
c_npu = torch.matmul(a_npu, b_npu)
torch.npu.synchronize()
t_npu = time.time() - t0
```

---

## 第 3 题（挑战）：Batch 矩阵乘法 vs 循环单次矩阵乘法

### 背景知识

**Batch 矩阵乘法（BMM）** 是指一次完成 B 个独立的矩阵乘法：

```
C[i] = A[i] × B[i],  i = 0, 1, ..., B-1
```

输入形状：`A: (B, M, K)`，`B: (B, K, N)`，输出形状：`C: (B, M, N)`。

**两种实现方式的对比**：

| 方式 | API | 调用次数 | NPU Kernel 启动次数 |
|:-----|:----|:---------|:-------------------|
| Batch | `torch.bmm(a, b)` | 1 次 | 1 次 |
| 循环 | `for i: torch.matmul(a[i], b[i])` | B 次 | B 次 |

每次 NPU Kernel 启动都有固定开销（任务下发、调度、同步），类似函数调用开销。当 B 很大时，循环方式的 Kernel 启动开销累加，导致性能显著下降。

### 详细答案

```python
import torch
import torch_npu
import time

batch_sizes = [8, 32, 128]
M, K, N = 256, 256, 256
reps = 20

print(f"{'B':>6} | {'bmm 耗时':>12} | {'循环 耗时':>12} | {'加速比':>8} | {'结果一致':>8}")
print("-" * 65)

for B in batch_sizes:
    # Step 1: 创建 (B, M, K) 和 (B, K, N) 随机矩阵，搬到 NPU
    a = torch.randn(B, M, K, dtype=torch.float32).npu()
    b = torch.randn(B, K, N, dtype=torch.float32).npu()

    # Step 2: Warmup
    _ = torch.bmm(a, b)
    torch.npu.synchronize()

    # Step 3: torch.bmm 计时
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(reps):
        c_bmm = torch.bmm(a, b)
    torch.npu.synchronize()
    t_bmm = (time.time() - t0) / reps * 1000

    # Step 4: 循环 torch.matmul 计时
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(reps):
        c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
    torch.npu.synchronize()
    t_loop = (time.time() - t0) / reps * 1000

    # Step 5: 验证结果一致
    consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)

    # Step 6: 打印结果
    speedup = t_loop / t_bmm
    print(f"{B:>6} | {t_bmm:>10.2f}ms | {t_loop:>10.2f}ms | {speedup:>7.1f}x | {'✓' if consistent else '✗':>8}")
```

**预期输出**（具体数值因环境而异）：
```
     8 |       0.04ms |       0.55ms |    12.5x |        ✓
    32 |       0.2 |       1.98ms |    45.3x |        ✓
   128 |       0.07ms |       7.71ms |   107.8x |        ✓
```

### 知识点详解

#### 1. `torch.stack` 的作用

```python
c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
```

列表推导式生成 B 个形状为 `(M, N)` 的矩阵乘结果，`torch.stack` 沿新维度（dim=0）拼成 `(B, M, N)` 的 batch 张量，与 `torch.bmm` 的输出形状一致。

```
a[0] @ b[0] → (M, N)     ┐
a[1] @ b[1] → (M, N)     ├─ torch.stack ─→ (B, M, N)
...                      │
a[B-1] @ b[B-1] → (M, N) ┘
```

#### 2. Kernel 启动开销

每次调用 `torch.matmul` 时，CANN 软件栈经历以下流程：

```
Python 调用 → torch_npu 适配 → CANN 算子库 → GE 图引擎 → Runtime 任务下发 → NPU Kernel 启动
```

这条链路有固定开销（约 0.05~0.1 ms/次），与矩阵大小无关。当 B=128 时，循环方式累积 128 次开销，而 `bmm` 只需 1 次。

#### 3. 思考题解答

**为什么 B 越大，循环方式越慢？**
- 每次循环都有一次 Kernel 启动开销（~0.05ms），B=128 时累积 ~6.4ms
- `bmm` 只启动 1 次 Kernel，开销固定 ~0.07ms

**什么情况下循环方式的劣势不明显？**
- 矩阵本身很大时（如 4096×4096），单次矩阵乘耗时远大于 Kernel 启动开销，开销占比小
- B 很小时（如 B=1），循环只调用 1 次，与 `bmm` 无差异

**在模型开发中如何避免不必要的循环调用？**
- 用 `torch.bmm` 替代循环 `torch.matmul`
- 用 `torch.vmap` 自动向量化
- 用算子融合（Operator Fusion）将多个算子合并为一个 Kernel

---

## 第 4 题（挑战）：在 NPU 上加速图像高斯模糊

### 背景知识

#### 高斯模糊

高斯模糊（Gaussian Blur）是计算机视觉中最基础的图像处理操作之一，用于降噪、平滑、预处理。其效果是让图片"变模糊"——中心像素的值被周围像素加权平均替代。

**数学定义**：二维高斯函数

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

其中 $\sigma$ 控制模糊程度：$\sigma$ 越大越模糊。

#### 卷积运算

高斯模糊的底层运算是**卷积**（Convolution）：用一个小矩阵（卷积核/滤波器）在图像上滑动，每个位置做逐元素乘加：

```
输出像素 = Σ (卷积核[i,j] × 输入像素[i,j])
```

对于 5×5 卷积核，每个输出像素需要 25 次乘加（MACs）。512×512 的 RGB 图片共有 512×512×3 = 786432 个输出像素，总计约 2000 万次乘加。

**卷积是 NPU Cube 单元的核心加速场景**——Cube 单元专为矩阵/卷积运算设计。

#### 分组卷积（groups=3）

RGB 图片有 3 个颜色通道。`groups=3` 表示每个通道独立卷积，不混合通道间信息。卷积核形状为 `[3, 1, 5, 5]`：3 组，每组 1 个输入通道，5×5 核。

```
R 通道 → 卷积核[0] → R 输出
G 通道 → 卷积核[1] → G 输出
B 通道 → 卷积核[2] → B 输出
```

### 详细答案

```python
import torch, torch_npu, torch.nn.functional as F
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, time

# Step 1: 加载图片并转为张量 [1, 3, H, W]
img = Image.open('./images/lena.jpg').convert('RGB')
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(f"图片张量形状: {img_tensor.shape}")

# 显示输入图片
plt.figure(figsize=(6, 5))
plt.imshow(img)
plt.title('Input: Lena (512x512)')
plt.axis('off')
plt.show()

# Step 2: 构造 5×5 高斯卷积核
size, sigma = 5, 1.0
coords = torch.arange(size, dtype=torch.float32) - size // 2  # [-2, -1, 0, 1, 2]
g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))            # 一维高斯
kernel2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)                 # 外积得到二维高斯
kernel2d = kernel2d / kernel2d.sum()                           # 归一化（权重之和=1）
weight = kernel2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1) # [3, 1, 5, 5]

# Step 3: CPU 卷积计时
start = time.time()
blur_cpu = F.conv2d(img_tensor, weight, padding=2, groups=3)
cpu_time = time.time() - start

# Step 4: NPU 卷积计时
img_npu = img_tensor.npu()
weight_npu = weight.npu()
torch.npu.synchronize()
start = time.time()
blur_npu = F.conv2d(img_npu, weight_npu, padding=2, groups=3)
torch.npu.synchronize()
npu_time = time.time() - start
result = blur_npu.cpu()

# Step 5: 打印耗时与加速比
print(f"CPU 卷积耗时: {cpu_time*1000:.2f} ms")
print(f"NPU 卷积耗时: {npu_time*1000:.2f} ms")
print(f"加速比: {cpu_time/npu_time:.1f}x")

# Step 6: 显示输出图片
blur_img = Image.fromarray((result.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
plt.figure(figsize=(6, 5))
plt.imshow(blur_img)
plt.title(f'Gaussian Blur (NPU, {npu_time*1000:.1f} ms)')
plt.axis('off')
plt.show()
```

**预期输出**（具体数值因环境而异）：
```
图片张量形状: torch.Size([1, 3, 512, 512])
CPU 卷积耗时: 6.77 ms
NPU 卷积耗时: 51.35 ms
加速比: 0.1x
```

### 知识点详解

#### 1. 图像转张量的维度变换

```python
img = Image.open('lena.jpg')          # PIL 图片: (H, W, C) = (512, 512, 3)
arr = np.array(img)                   # numpy: (H, W, C)
t = torch.from_numpy(arr)             # tensor: (H, W, C)
t = t.permute(2, 0, 1)                # tensor: (C, H, W) = (3, 512, 512)  ← PyTorch 格式
t = t.unsqueeze(0)                    # tensor: (1, C, H, W) = (1, 3, 512, 512)  ← 加 batch 维
t = t.float() / 255.0                 # 归一化到 [0, 1]
```

PyTorch 卷积层要求输入格式为 **NCHW**：`N`=batch, `C`=channels, `H`=height, `W`=width。

#### 2. 高斯核的构造

```python
coords = torch.arange(5) - 2          # [-2, -1, 0, 1, 2]
g1d = torch.exp(-(coords**2) / 2)     # 一维高斯: [0.135, 0.607, 1.000, 0.607, 0.135]
kernel2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # 外积 → 5×5 二维高斯
kernel2d = kernel2d / kernel2d.sum()  # 归一化
```

二维高斯可分解为两个一维高斯的外积（乘积），这叫**可分离卷积**（Separable Convolution）。构造出的 5×5 核：

```
[[0.003, 0.013, 0.022, 0.013, 0.003],
 [0.013, 0.060, 0.098, 0.060, 0.013],
 [0.022, 0.098, 0.162, 0.098, 0.022],
 [0.013, 0.060, 0.098, 0.060, 0.013],
 [0.003, 0.013, 0.022, 0.013, 0.003]]
```

中心权重最大（0.162），四周逐渐衰减，符合高斯分布的"钟形"特征。

#### 3. `padding=2` 的作用

5×5 卷积核的中心在第 3 个位置。如果不 padding，输出尺寸会缩小：`512 - 5 + 1 = 508`。`padding=2` 在图像四周补 2 圈零，使输出尺寸不变：`512 + 2×2 - 5 + 1 = 512`。

#### 4. 为什么单张图片 NPU 反而更慢？

单张 512×512 图片的卷积计算量很小（~20M FLOPs），但数据搬运需要把图片和卷积核从 CPU 搬到 NPU（~3MB），再把结果搬回（~3MB）。

**算术强度分析**：

**矩阵乘法 4096² 的 FLOPs 与 Bytes**：
- FLOPs = `2 × 4096³` ≈ 137G
- Bytes = `3 × 4096² × 4` ≈ 201MB
- 算术强度 = `137G / 201MB` ≈ **683 FLOPs/byte**

**5×5 卷积的 FLOPs 与 Bytes**（按单个输出像素计算）：
- FLOPs：每个输出像素需要 5×5=25 次乘加 = `2 × 25` = 50 FLOPs
- Bytes：读 5×5=25 个输入像素 + 写 1 个输出像素 = `(25 + 1) × 4` = 104 Bytes
- 算术强度 = `50 / 104` ≈ **0.48 FLOPs/byte**

| | 矩阵乘法 4096² | 5×5 卷积 |
|:---|:---:|:---:|
| FLOPs | 2×4096³ ≈ 137G | 2×25 = 50（每像素） |
| Bytes | 3×4096²×4 ≈ 201MB | (25+1)×4 = 104（每像素） |
| 算术强度 | ~683 FLOPs/byte | ~0.48 FLOPs/byte |
| 瓶颈 | 计算密集 (Compute-bound) | 访存密集 (Memory-bound) |
| 加速比 | ~300x+ | ~0.1x（单张） |

**直观类比**：
- **矩阵乘法**：搬 1 趟砖，砌 683 块墙 → NPU 算力优势充分发挥
- **5×5 卷积**：搬 1 趟砖，只砌 0.48 块墙 → 大部分时间花在搬运上

**批量处理时**（32 张图片），计算量增加 32 倍但搬运只增加 1 次（批量搬运），加速比可提升到 ~30x。

> **真实 CNN 中的情况**：ResNet 等网络中卷积通道数达 64~256，算术强度大幅提升（从 0.48 到 10~50+），NPU 加速比通常可达 **50x~200x**。本教程的 3 通道 RGB 图像只是最简单的示例。

#### 5. `F.conv2d` 参数详解

```python
F.conv2d(input, weight, bias=None, stride=1, padding=0, groups=3)
```

| 参数 | 本题值 | 含义 |
|:-----|:-------|:-----|
| `input` | `(1, 3, 512, 512)` | 输入张量 NCHW |
| `weight` | `(3, 1, 5, 5)` | 卷积核：(out_channels, in_channels/groups, kH, kW) |
| `padding` | `2` | 四周补零圈数，保持输出尺寸 |
| `groups` | `3` | 分组卷积，3 个通道各自独立卷积 |

---

## 总结：四道题的知识脉络

```
第 1 题：张量 → .npu() → 逐元素运算 → .cpu() 验证
         ↓
第 2 题：矩阵乘法 → CPU vs NPU 性能对比 → 异步同步
         ↓
第 3 题：Batch 运算 → bmm vs 循环 → Kernel 启动开销
         ↓
第 4 题：卷积运算 → 图像处理 → 算术强度分析
```

**核心收获**：
1. **`.npu()` / `.cpu()`** 是异构计算的基本操作——数据搬运
2. **`synchronize`** 是 NPU 异步计时的必备操作——等待计算完成
3. **算术强度** 决定加速比上限——计算密集型获益大，访存密集型获益小
4. **Batch 运算** 优于循环调用——减少 Kernel 启动开销
5. **Cube 单元** 擅长矩阵/卷积，**Vector 单元** 擅长逐元素运算
