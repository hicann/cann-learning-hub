# Lab5_2 部署实验 —— 昇腾香橙派 310B

本目录使用 PyTorch + `torch_npu` 在昇腾香橙派 310B（Ascend 310B）上验证四个阶段的向量算子，对应 Ascend C 算子开发的 Host 侧功能验证。

## 环境依赖

- Python 3
- PyTorch
- `torch_npu`（昇腾 NPU 适配包）
- Ascend Toolkit 环境（脚本中会自动 `source set_env.sh`）

## 文件说明

### `add_8core_orangepi.py` —— 阶段一：双向量加法（8 核）

- 算子：`z = x + y`
- 输入 shape：`(8, 2048)`，共 16384 个元素
- 输入 `x` 全填 1.2，`y` 全填 2.3，预期输出全为 3.5
- 8 个 AI Core 并行，每核处理 2048 个元素
- 运行：`python3 add_8core_orangepi.py`

### `add_32core_orangepi.py` —— 阶段二：双向量加法（32 核）

- 算子：`z = x + y`
- 输入 shape：`(32, 2048)`，共 65536 个元素
- 输入 `x` 全填 2.2，`y` 全填 2.3，预期输出全为 4.5
- 在阶段一基础上核数 8→32、数据量 16384→65536，每核处理量不变
- 运行：`python3 add_32core_orangepi.py`

### `add3_orangepi.py` —— 阶段三：三向量加法

- 算子：`z = x + y + w`
- 输入 shape：`(8, 2048)`，共 16384 个元素
- 输入 `x` 全填 1.2、`y` 全填 2.3、`w` 全填 3.4，预期输出全为 6.9
- 在阶段一基础上增加第三个输入向量，底层分两次 Add 指令执行
- 运行：`python3 add3_orangepi.py`

### `mul_orangepi.py` —— 阶段四：双向量乘法

- 算子：`z = x * y`
- 输入 shape：`(8, 2048)`，共 16384 个元素
- 输入 `x` 全填 1.2、`y` 全填 2.3，预期输出全为 2.76
- 在阶段一基础上将 Add 指令替换为 Mul 指令
- 运行：`python3 mul_orangepi.py`

### `run_orangepi.sh` —— 一键运行脚本

按阶段选择运行对应脚本，默认运行全部（`all`）。

```bash
bash run_orangepi.sh        # 运行全部阶段
bash run_orangepi.sh 1      # 阶段一：加法 8 核
bash run_orangepi.sh 2      # 阶段二：加法 32 核
bash run_orangepi.sh 3      # 阶段三：三向量加法
bash run_orangepi.sh 4      # 阶段四：乘法
```

## 执行流程（每个 Python 脚本通用）

1. **数据参数定义**：核数、每核数据量、填充值
2. **NPU 设备初始化**：`torch.device('npu:0')`
3. **Host 侧生成输入数据**：`torch.full`
4. **数据搬移到 NPU**（Host → Device）：`.to(device)`
5. **NPU 上执行算子计算**
6. **结果搬回 Host**（Device → Host）：`.cpu()`
7. **Golden 计算与精度验证**：`torch.allclose` 逐元素比对

## 预期结果

每个脚本精度验证通过时输出：

```
[Success] 精度验证通过！
```
