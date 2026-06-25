# CANNJudge 算子提交 Skill

帮助用户完成 CANNJudge 算子竞赛的完整流程：下载工程、理解题目、实现泛化算子、提交代码、获取结果。

## 触发条件

当用户需要：
- 参加 CANNJudge 算子竞赛
- 从 CANNJudge 下载算子工程模板
- 提交算子实现到 CANNJudge
- 查看 CANNJudge 排行榜

## 前置要求

- 已注册 CANNJudge 账号
- 已配置 CANN 开发环境
- 已安装必要的编译工具
- 已安装 pycryptodome：`pip install pycryptodome`

## 安全登录机制（RSA 加密）

### 铁律

1. **禁止在对话中直接输入明文密码**
2. **禁止将解密后的密码显示/打印/输出到对话中** — 解密仅用于内部调用登录 API，绝不出现在任何输出中
3. **禁止将明文密码写入日志、文件或环境变量**

违反以上任何一条视为严重安全事件。

### 原理

```
服务器 (private.pem)          个人 PC (public.pem)
       │                            │
       │◄─── 拷贝 public.pem ───────┤
       │                            │
       │         密码 ──► RSA加密 ──┤──► 密文
       │                            │
       ├── RSA解密(密文) ──► 明文   │  ← 仅内存中，禁止输出
       ├── 登录 CANNJudge           │
       ├── 明文立即从内存清除        │
```

- **私钥 (private.pem)**：仅存在于服务器，用于解密，**绝不外传**
- **公钥 (public.pem)**：可公开分发，用户在个人 PC 上用于加密密码
- **密文**：用户将加密后的密文提供给 CANNBot，CANNBot 在服务器上解密后登录
- **明文密码**：仅在内存中短暂存在，用于调用登录 API，**禁止输出到对话/日志/文件**

### 首次设置（一次性）

#### 步骤 1：在服务器上生成密钥对

```bash
# 在服务器上运行
python3 skills/cannjudge-submit/generate_key.py
# 生成 private.pem（留在服务器）和 public.pem（拷贝到 PC）
```

#### 步骤 2：将公钥拷贝到个人 PC

```bash
# 方式 A：直接复制服务器上 public.pem 的内容
cat public.pem  # 在服务器上查看，复制内容到 PC

# 方式 B：用 scp 拷贝
scp server:~/path/to/public.pem ~/local/path/
```

#### 步骤 3：在个人 PC 上加密密码

```bash
# 在个人 PC 上运行
python3 encrypt_password.py --public-key public.pem
# 输入密码后，输出 RSA 密文
```

#### 步骤 4：将密文提供给 CANNBot

将步骤 3 输出的密文直接粘贴给 CANNBot，CANNBot 会用服务器上的私钥解密并登录。

### 日常使用

设置完成后，每次登录只需：
1. 在 PC 上用 `encrypt_password.py` 加密密码（或复用之前的密文，只要私钥不变）
2. 将密文给 CANNBot

**密文可以复用**：只要私钥文件不变，同一密文解密结果相同，无需每次重新加密。

## ⚠️ 重要：必须从网站获取题目信息

**禁止使用本地任何算子信息文件，也不要私自决定使用标准定义！**

所有题目信息必须从 CANNJudge 网站获取：
1. **题目描述**：通过 API 或网页获取
2. **算子原型**：从题目描述中解析
3. **工程模板**：通过 API 下载

**原因**：
- 题目可能随时更新
- 确保信息准确性和一致性
- 竞赛存在限制

**获取方式**：
```python
# 方式1: 通过 API 获取（推荐）
resp = requests.get(f"https://cannjudge.cn/api/problems/{problem_id}")
problem_info = resp.json()

# 方式2: 通过网页获取
# 使用 webfetch 工具获取题目页面
```

## ⚠️ 重要：理解题目和参考算子

**必须深入理解题目描述和参考算子的行为！**

### 关键步骤

1. **仔细阅读题目描述**
   - 算子原型（输入/输出/属性）
   - 支持的数据类型
   - 数学公式和计算规则
   - **参考算子**（如 PyTorch、TensorFlow）

2. **查阅参考算子的官方文档**（最重要！）
   - 题目通常会指定参考算子（如 `torch.histc`）
   - **必须查阅官方文档**，理解所有行为
   - 特别注意：**默认值的特殊语义**
   - 注意：边界情况、特殊参数值

3. **理解默认值的真实含义**
   - 默认值不一定表示"使用这个值"
   - 可能是一个**标志**，表示：
     - "需要自动计算"
     - "使用数据的实际范围"
     - "忽略此参数"
   - **示例**：`torch.histc(min=0, max=0)` 表示自动计算数据范围

### 常见陷阱

| 陷阱 | 错误理解 | 正确理解 |
|------|---------|---------|
| `min=0, max=0` | 范围是 [0, 0] | 自动计算数据的 min 和 max |
| 默认值 | 直接使用默认值 | 可能是特殊标志 |
| 参考算子 | 仅参考原型 | 必须完全复现所有行为 |

### 实战案例：Histogram 算子

**题目描述**：
- 参考算子：`torch.histc`
- 属性：`bins=100, min=0.0, max=0.0`（默认值）

**错误实现**：
```cpp
// ❌ 错误：认为 min=0.0, max=0.0 是有效范围
if (min_val >= max_val) {
    min_val = 0.0f;
    max_val = 1.0f;  // 使用默认范围
}
```

**正确实现**：
```cpp
// ✅ 正确：查阅 torch.histc 文档
// 当 min=0 且 max=0 时，自动计算数据的 min 和 max
if (min_val == 0.0f && max_val == 0.0f) {
    // 需要在 Kernel 中动态计算数据范围
    need_compute_range = true;
}
```

**教训**：
- ❌ 没有查阅 PyTorch 文档
- ❌ 错误理解默认值的含义
- ✅ 查阅文档后发现特殊语义
- ✅ 实现动态计算数据范围

### 检查清单

在开始实现前，必须完成：

- [ ] **阅读题目描述**：理解算子原型和数学公式
- [ ] **查阅参考算子文档**：理解所有行为和特殊语义
- [ ] **理解默认值**：是否有特殊含义
- [ ] **理解边界情况**：特殊输入、特殊属性值
- [ ] **验证理解**：用参考算子测试边界情况（如可能）

---

## ⚠️ 重要：算子泛化性设计

**平台不开放测试用例 API，必须设计泛化算子！**

### 为什么需要泛化设计

1. **无法获取测试用例**：平台不提供测试用例 API，无法提前知道具体的 shape、dtype
2. **测试用例多样**：平台会使用多种 shape、dtype、属性组合进行测试
3. **一劳永逸**：泛化算子可以处理所有可能的输入情况

### 泛化设计核心要素

| 要素 | 说明 | 设计要点 |
|------|------|----------|
| **Shape 泛化** | 支持任意维度和大小 | 使用动态 shape 处理，不要硬编码维度数 |
| **Dtype 泛化** | 支持多种数据类型 | 使用模板类，根据 dtype 参数选择实现 |
| **属性泛化** | 支持属性的各种取值 | 处理边界值、默认值、特殊值 |
| **对齐泛化** | 处理对齐和非对齐情况 | 使用 DataCopyPad，处理尾部数据 |
| **边界泛化** | 处理边界情况 | 空输入、单元素、极端值等 |

### 泛化设计检查清单

**Shape 泛化**：
- [ ] 支持 1D、2D、3D、4D 等任意维度
- [ ] 支持任意大小（大、中、小）
- [ ] InferShape 正确处理动态 shape
- [ ] Tiling 函数不依赖固定维度

**Dtype 泛化**：
- [ ] 支持题目要求的所有 dtype
- [ ] 使用模板类实现
- [ ] Float16 使用 Float32 中间计算
- [ ] 正确处理精度转换

**属性泛化**：
- [ ] 处理所有可能的属性值
- [ ] 处理默认值
- [ ] 处理边界值（如 axis=-1）
- [ ] 处理特殊值（如 keepdims=True/False）

**对齐泛化**：
- [ ] 处理 32 字节对齐
- [ ] 处理非对齐的尾部数据
- [ ] 使用 DataCopyPad 自动处理

**边界泛化**：
- [ ] 空输入（shape 包含 0）
- [ ] 单元素输入
- [ ] 极端值（最大、最小）
- [ ] 特殊值（NaN、Inf）

### 泛化设计示例

```cpp
// ❌ 错误：硬编码维度
int32_t N = shape.GetDim(0);
int32_t C = shape.GetDim(1);
int32_t H = shape.GetDim(2);
int32_t W = shape.GetDim(3);

// ✅ 正确：动态处理任意维度
int32_t rank = shape.GetDimNum();
uint32_t totalLength = 1;
for (int32_t i = 0; i < rank; i++) {
    totalLength *= shape.GetDim(i);
}
```

```cpp
// ❌ 错误：只支持 float
void Process(GM_ADDR x, GM_ADDR y) {
    AscendC::GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer((__gm__ float *)x);
    // ...
}

// ✅ 正确：模板支持多 dtype
template<typename T>
class KernelOp {
    void Process(GM_ADDR x, GM_ADDR y) {
        AscendC::GlobalTensor<T> xGm;
        xGm.SetGlobalBuffer((__gm__ T *)x);
        // ...
    }
};

// Kernel 入口根据 dtype 选择模板
extern "C" __global__ __aicore__ void op_kernel(...) {
    if (dtype == 0) {      // DT_FLOAT
        KernelOp<float> op;
        op.Process(...);
    } else if (dtype == 1) { // DT_FLOAT16
        KernelOp<half> op;
        op.Process(...);
    }
}
```

```cpp
// ❌ 错误：假设数据对齐
uint32_t tileNum = totalLength / 256;

// ✅ 正确：处理非对齐情况
uint32_t tileNum = (totalLength + 255) / 256;
uint32_t lastTileLength = totalLength - (tileNum - 1) * 256;
```

## API 接口

### 基础 URL
```
https://cannjudge.cn
```

### 用户相关
- **登录**: `POST /api/users/login`
  - 请求体: `{"email": "用户邮箱", "password": "密码"}`
  - 返回: 用户信息，包含 `_id` (用户ID)

### 题目相关
- **获取题目信息（通过ID）**: `GET /api/problems/{problemId}`
  - 返回题目详情，包含名称、描述等

- **获取题目信息（通过名称）**: `GET /api/problems/name/{problemName}`
  - **推荐使用此API**
  - 返回题目详情，包含名称、描述等
  - 示例: `GET /api/problems/name/depthtospace`

- **下载工程模板**: `GET /api/problems/{problemId}/package?userId={userId}`
  - 返回 Zip 工程包

### 提交相关
- **提交代码**: `POST /api/submissions/submit`
  - 请求体:
    ```json
    {
      "problemId": "题目ID",
      "userId": "用户ID",
      "tiling_h": "tiling头文件内容",
      "tiling_key_h": "tiling key头文件内容",
      "host_cpp": "host侧cpp内容",
      "kernel_cpp": "kernel侧cpp内容"
    }
    ```
  - 返回: `{"code": 0, "data": {"submissionId": "提交ID"}}`

- **查询提交结果**: `GET /api/submissions/{submissionId}`
  - 返回提交状态、测试结果、精度比例等

- **获取排行榜**: `GET /api/submissions/problem/{problemId}/latest`
  - 返回该题目的所有用户最新提交

## 完整流程

### 步骤 1: 登录

**推荐方式：RSA 加密密文登录（安全）**

```python
import requests
from cannjudge_cli import CANNJudgeClient

client = CANNJudgeClient()

# 方式 A：用密文登录（推荐）
# 密文由用户在 PC 上用 encrypt_password.py 生成
# 解密后的明文仅在内存中用于调用登录 API，禁止输出到对话/日志/文件
user_info = client.login_with_ciphertext(
    email="用户邮箱",
    ciphertext="RSA加密后的密文",
    private_key_path="private.pem"  # 服务器上的私钥路径
)

# 方式 B：直接用明文密码登录（不推荐，仅调试用）
# user_info = client.login("用户邮箱", "明文密码")
```

**命令行方式**：

```bash
# 推荐：密文登录
python cannjudge_cli.py login --email "your@email.com" --ciphertext "RSA密文"

# 不推荐：明文登录
python cannjudge_cli.py login --email "your@email.com" --password "明文密码"
```

**安全铁律**：登录成功后只输出"登录成功"和用户 ID，**绝不输出解密后的密码**。

### 步骤 2: 获取题目信息

```python
# 方式1: 如果知道题目ID
problem_id = "题目ID"
resp = requests.get(
    f"https://cannjudge.cn/api/problems/{problem_id}",
    cookies=cookies
)
problem_info = resp.json()

# 方式2: 通过题目名称查找（需要先获取题目列表）
# 注意: 题目列表 API 可能需要从赛事/小组入口获取
```

### 步骤 3: 下载工程模板

```python
# 下载工程包
resp = requests.get(
    f"https://cannjudge.cn/api/problems/{problem_id}/package",
    params={"userId": user_id},
    cookies=cookies
)

# 保存并解压
with open("project.zip", "wb") as f:
    f.write(resp.content)

import zipfile
with zipfile.ZipFile("project.zip", "r") as zf:
    zf.extractall("project")
```

### 步骤 4: 理解题目

从题目信息中获取：
- `name`: 算子名称
- `title`: 算子标题
- `desc`: 题目描述（Markdown格式）

**重要**：仔细阅读题目描述，理解：
1. 算子原型（输入/输出/属性）
2. 支持的数据类型
3. 支持的 shape 维度
4. 特殊要求和限制

### 步骤 5: 设计泛化算子

**⚠️ 核心步骤：根据 `ascendc-ops-project` skill 设计泛化算子**

`ascendc-ops-project` 提供完整的泛化算子开发指导：
- **Shape 泛化**：动态 shape 处理方法
- **Dtype 泛化**：模板类实现方法
- **Tiling 设计**：多核切分、UB 切分
- **Host 实现**：TilingFunc、InferShape、InferDataType
- **Kernel 实现**：模板类、多核处理
- **API 最佳实践**：避免常见错误
- **精度验证标准**：确保结果正确

工程模板结构：
```
code/
├── CMakeLists.txt           # 主 CMake 配置
├── op_host/
│   ├── CMakeLists.txt
│   └── {op_name}.cpp        # Host 侧实现（Tiling、InferShape）
└── op_kernel/
    ├── CMakeLists.txt
    ├── {op_name}_tiling.h   # Tiling 数据结构
    ├── tiling_key_{op_name}.h  # Tiling Key 定义
    └── {op_name}.cpp        # Kernel 侧实现
```

需要实现的文件：
1. **op_kernel/{op_name}_tiling.h**: 定义 Tiling 数据结构（包含泛化参数）
2. **op_host/{op_name}.cpp**: 实现泛化的 TilingFunc、InferShape、InferDataType
3. **op_kernel/{op_name}.cpp**: 实现泛化的算子计算逻辑

**设计建议**：
1. **先分析泛化需求**：
   - 题目要求支持哪些 dtype？
   - 是否需要支持任意维度？
   - 有哪些属性需要处理？
   - 是否有特殊边界情况？

2. **设计泛化 Tiling**：
   - Tiling 数据结构包含所有泛化参数
   - 不硬编码 shape 维度
   - 正确处理对齐和非对齐

3. **实现泛化 Host**：
   - InferShape 支持动态 shape
   - TilingFunc 处理所有 dtype
   - 正确读取属性和 ValueDepend

4. **实现泛化 Kernel**：
   - 使用模板类支持多 dtype
   - 正确处理多核切分
   - 处理边界和对齐情况

5. **验证泛化性**：
   - 检查是否支持所有要求的 dtype
   - 检查是否支持任意维度
   - 检查是否处理边界情况
   - 参考 `ascendc-ops-project` 的检查清单

### 步骤 6: 编译工程

```bash
# 需要添加编译配置文件（如果模板中没有）
# CMakePresets.json 和 build.sh 可以从已有工程复制

cd project/code
bash build.sh
```

### 步骤 7: 提交代码

```python
# 读取实现的代码文件
with open("op_kernel/{op_name}_tiling.h") as f:
    tiling_h = f.read()
with open("op_kernel/tiling_key_{op_name}.h") as f:
    tiling_key_h = f.read()
with open("op_host/{op_name}.cpp") as f:
    host_cpp = f.read()
with open("op_kernel/{op_name}.cpp") as f:
    kernel_cpp = f.read()

# 提交
resp = requests.post(
    "https://cannjudge.cn/api/submissions/submit",
    json={
        "problemId": problem_id,
        "userId": user_id,
        "tiling_h": tiling_h,
        "tiling_key_h": tiling_key_h,
        "host_cpp": host_cpp,
        "kernel_cpp": kernel_cpp
    },
    cookies=cookies
)

result = resp.json()
submission_id = result["data"]["submissionId"]
```

### 步骤 8: 查询结果

```python
import time

# 轮询等待结果
for i in range(30):
    time.sleep(3)
    
    resp = requests.get(
        f"https://cannjudge.cn/api/submissions/{submission_id}",
        cookies=cookies
    )
    
    result = resp.json()
    status = result.get("status")
    
    if status in ["Accepted", "Wrong Answer", "Compile Error", "Runtime Error"]:
        print(f"最终状态: {status}")
        
        # 打印每个测试用例结果
        for tc in result.get("result", []):
            tc_status = tc.get("testcase_status")
            precision = tc.get("precision_ratio", "N/A")
            time_ms = tc.get("time", "N/A")
            print(f"  {tc_status} (precision: {precision}, time: {time_ms}ms)")
        
        break
```

### 步骤 9: 查看排行榜

```python
resp = requests.get(
    f"https://cannjudge.cn/api/submissions/problem/{problem_id}/latest",
    cookies=cookies
)

rankings = resp.json()
# 按 Accepted 状态和分数排序
```

## 常见问题

### 1. 为什么我的算子在部分测试用例上失败？

**原因**：算子不够泛化，无法处理某些特殊情况。

**解决方法**：
1. 查看失败的测试用例信息（如果有）
2. 检查算子是否处理了所有泛化场景
3. 参考 `ascendc-ops-project` 的泛化设计检查清单
4. 添加对边界情况的处理

### 2. 如何设计支持任意维度的算子？

**方法**：
```cpp
// Host 侧：动态处理 shape
int32_t rank = shape.GetDimNum();
uint32_t totalLength = 1;
for (int32_t i = 0; i < rank; i++) {
    totalLength *= shape.GetDim(i);
}

// Kernel 侧：按总长度处理，不依赖具体维度
for (uint32_t i = 0; i < totalLength; i++) {
    // 处理每个元素
}
```

### 3. 如何处理 axis 属性的负值？

**方法**：
```cpp
// Host 侧：标准化 axis
int32_t axis = attrs->GetInt(0);
int32_t rank = shape.GetDimNum();
if (axis < 0) {
    axis = axis + rank;  // 转换为正值
}
```

### 4. 精度要求是什么？

- float16: rtol < 1e-3, atol < 1e-3
- float32: rtol < 1e-4, atol < 1e-4

### 5. 编译配置文件缺失怎么办？

如果下载的模板缺少编译配置文件：
- `CMakePresets.json`: 编译预设配置
- `build.sh`: 编译脚本

可以从已有的 Ascend C 工程复制这些文件。

### 6. 提交字段说明

- `tiling_h`: Tiling 数据结构定义（.h 文件内容）
- `tiling_key_h`: Tiling Key 模板定义（.h 文件内容）
- `host_cpp`: Host 侧实现（.cpp 文件内容）
- `kernel_cpp`: Kernel 侧实现（.cpp 文件内容）

**注意**: 如果某些文件不存在（如 tiling_key_h），提交空字符串即可。

## 返回结果说明

### 提交状态
- `Running`: 正在执行
- `Accepted`: 全部通过
- `Wrong Answer`: 部分测试用例失败
- `Compile Error`: 编译失败
- `Runtime Error`: 运行时错误
- `Time Limit Exceeded`: 超时

### 测试用例结果
- `testcase_status`: 该测试用例状态
- `precision_ratio`: 精度比例（1.0 表示完全匹配）
- `time`: 执行时间（毫秒）

## 注意事项

1. **密码安全**: 禁止在对话中直接输入明文密码，必须使用 RSA 加密密文登录
2. **私钥保护**: private.pem 仅保留在服务器，绝不外传、不提交到代码仓库
3. **泛化设计**: 必须设计泛化算子，平台不提供测试用例
4. **Cookie 管理**: 登录后保存 cookie，用于后续所有请求
5. **轮询间隔**: 查询结果时建议间隔 3 秒，避免频繁请求
6. **超时处理**: 设置合理的请求超时时间（建议 30 秒）

## 示例用法

用户可以这样使用此 skill：

```
用户: 帮我从 CANNJudge 下载 DepthToSpace 题目并实现泛化算子

助手: 
1. 请提供 CANNJudge 登录邮箱和 RSA 加密密文
   （如未设置，先运行 generate_key.py 生成密钥对，再在 PC 上用 encrypt_password.py 加密密码）
2. 我将用服务器私钥解密密文，登录 CANNJudge
3. 获取题目信息
4. 下载工程模板
5. 分析题目要求，设计泛化算子：
   - 支持的数据类型：float16, float32
   - 支持的维度：4D 输入
   - 属性：block_size
   - 边界情况：block_size=1, block_size=最大值
6. 实现泛化的 Host 和 Kernel 代码
7. 编译并提交
8. 查看结果和排行榜
```

## 相关 Skills

**核心依赖**：
- `ascendc-ops-project`: Ascend C 算子工程化开发完整指南
  - 包含：工程创建、泛化设计、Tiling 设计、Host/Kernel 实现、API 最佳实践、精度验证标准
  - 用途：提供泛化算子实现的所有技术细节

**使用流程**：
1. 使用本 skill（cannjudge-submit）处理 CANNJudge 平台交互（登录、下载、提交、查询）
2. 使用 `ascendc-ops-project` skill 指导泛化算子设计和实现
3. 结合两个 skill 完成完整的竞赛流程

---

## FAQ

### Q: float16 数据精度不达标怎么办？

**A**: float16 精度限制可能导致边界量化问题，需要根据参考算子的行为调整实现。

**常见问题**：
- numpy/torch 在 float16 数据上会将边界量化到 float16 可表示值
- 这导致边界不均匀，与均匀的 binWidth 计算不一致
- 当值恰好等于边界时，bin 索引计算错误

**解决方案**：
1. **理解参考算子行为**：用 Python 测试参考算子在 float16 数据上的精确行为
2. **根据数据类型调整**：
   - float16: 量化边界到 float16 可表示值
   - float32/int32: 使用均匀边界
3. **实现技巧**：
   - 使用二分查找计算 bin 索引
   - 对 float16 数据，将边界转换为 half 再转回 float
   - 在 host 侧设置标志，kernel 中根据标志选择行为

**示例代码**：
```cpp
// Host 侧：根据 dtype 设置量化标志
int32_t quantize_edges = (dtype_x == ge::DT_FLOAT16) ? 1 : 0;
tiling->quantizeEdges = quantize_edges;

// Kernel 侧：根据标志选择是否量化边界
if (quantizeEdges) {
    half edge_h = static_cast<half>(edge);
    edge = static_cast<float>(edge_h);
}
```

### Q: 如何处理参考算子的默认值特殊语义？

**A**: 默认值可能不是实际值，而是特殊标志。

**常见模式**：
- `min=0, max=0` → 自动计算数据范围
- `axis=-1` → 使用最后一个维度
- `keepdims=False` → 不保持维度

**解决方法**：
1. 查阅参考算子官方文档
2. 用 Python 测试边界情况
3. 在代码中正确处理特殊语义

### Q: 如何设计泛化算子？

**A**: 参考 `ascendc-ops-project` skill 的泛化设计指南。

**核心要素**：
- Shape 泛化：动态处理任意维度
- Dtype 泛化：模板支持多数据类型
- 属性泛化：处理所有可能的属性值
- 对齐泛化：使用 DataCopyPad 处理非对齐数据
- 边界泛化：处理空输入、单元素、极端值等
