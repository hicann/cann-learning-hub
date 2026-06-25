# as_strided 算子

## 基本信息

- **算子名称**：as_strided
- **算子类别**：Conversion（Index/Gather）
- **支持数据类型**：float16, float32, int32
- **支持芯片**：ascend910b
- **创建时间**：2026-05-06
- **开发状态**：已完成

## 数学公式

```
输入:  input_x - 任意形状, dtype ∈ {float16, float32, int32}
属性:  size[n]    - 输出张量各维度大小
       stride[n]  - 输出张量各维度步长（可为正/负/零）
       storage_offset - 起始偏移量（>= 0）
输出:  output - shape = size, dtype 与 input_x 相同

数学公式:
output[i_0, i_1, ..., i_{n-1}] = input_x[storage_offset + Σ(i_j × stride_j)]

其中 i_j ∈ [0, size_j)，j = 0, 1, ..., n-1
```

## 目录结构

```
as_strided/
├── code/
│   ├── op_kernel/        # Kernel 侧代码
│   │   ├── as_strided.cpp         # Kernel 计算逻辑
│   │   ├── as_strided_tiling.h    # Tiling 结构体
│   │   └── tiling_key_as_strided.h # TilingKey 定义
│   ├── op_host/          # Host 侧代码
│   │   └── as_strided.cpp         # Host Tiling 逻辑
│   └── build/            # 编译输出
├── scripts/
│   ├── golden.py         # Golden 参考实现
│   └── gen_data.py       # 测试数据生成
├── docs/                 # 设计、计划、环境、审查文档
├── run_test.py           # 精度验证脚本
└── README.md
```

## 编译与运行

### 编译

```bash
cd code/build
cmake ..
make -j
```

### 安装

```bash
bash code/build/scripts/install.sh
```

### 测试

```bash
python3 run_test.py
```

## API 映射

| 数学操作 | Ascend C API | 说明 |
|---------|-------------|------|
| 输入数据 GM→UB 搬运 | DataCopyPad | 全量输入搬运（Path A）或单元素搬运（Path B） |
| 按 offset 表收集元素 | Gather | Path A: UB 内按字节偏移收集，硬件指令 |
| 偏移表 GM→UB 搬运 | DataCopyPad | 从 tiling data 搬运 Host 预计算的偏移表 |
| 输出数据 UB→GM 搬运 | DataCopyPad | 非对齐写回，自动处理 padding |
| 逐元素写入（Path B 回退） | GetValue + SetValue | Path B 回退方案，性能不作为验收标准 |

### 路径选择

- **Path A**（主路径）：输入全量入 UB + Gather 收集。当 `inputTotalBytes + offsetTableBytes + outputTileBytes×2 ≤ UBBudget` 时选择。
- **Path B**（回退路径）：逐元素 DataCopyPad 读取。当输入过大无法全量入 UB 时选择。

### 偏移表预计算策略

- **优先方案**：Host 侧预计算偏移表，追加到 tiling data。Kernel 通过 DataCopyPad 从 tiling GM 搬运到 UB，消除 SetValue 循环和除法/取模运算。
- **回退方案**：当偏移表超过 tiling data 容量时，Kernel 内使用增量索引计算（混合进制计数器），仅使用加法/减法/比较，无除法/取模。

## 已知限制

1. **Path B 性能**：Path B 使用 GetValue+SetValue 逐元素操作，为已知反模式。仅作为大输入场景的回退方案，性能不作为验收标准。
2. **负 stride clamp 语义**：负 stride 导致源索引越界时，clamp 到 input[0]，输出值可能与 PyTorch 语义不同（PyTorch 不支持负 stride）。
3. **BF16 不支持**：当前实现不支持 bfloat16 数据类型。
4. **偏移表容量限制**：当输出元素数超过 tiling data 容量时，回退到 Kernel 内增量计算（仍使用 SetValue 构建 Gather 偏移表），性能有所折衷。

## 精度标准

| 数据类型 | rtol | atol | 说明 |
|---------|------|------|------|
| float32 | 1e-5 | 1e-5 | 单精度浮点 |
| float16 | 1e-3 | 1e-3 | 半精度浮点 |
| int32 | 0 | 0 | 完全准确（bitwise match） |

## 参考资料

- 体验报告：README.md
- 开发教程：docs/TUTORIAL.md
- 设计文档：docs/DESIGN.md
- 设计串讲：docs/WALKTHROUGH.md
- 代码审查：docs/REVIEW.md
- 开发计划：docs/PLAN.md
- 环境检查：docs/environment.json
